//
// Created by qzz on 2024/1/23.
//

#ifndef BRIDGE_ACTOR_H
#define BRIDGE_ACTOR_H
#include <memory>
#include "bridge_lib/bridge_utils.h"
#include "rela/batch_runner.h"
#include "rela/batcher.h"
#include "rela/prioritized_replay.h"
#include "rela/tensor_dict.h"
#include "rela/types.h"

#include "rlcc/env.h"
#include "rlcc/rnn_buffer.h"

namespace rela {
// A transition which only stores state, action and reward, as described in Douzero paper.
class SARTransition {
 public:
  SARTransition() = default;

  SARTransition(TensorDict &obs, TensorDict &action, torch::Tensor &reward,
                torch::Tensor &terminal)
      : obs(obs), action(action), reward(reward), terminal(terminal) {}

  SARTransition index(int i) const;
  TensorDict obs;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
};
}  // namespace rela

namespace rlcc {

namespace ble = bridge_learning_env;

class Actor {
 public:
  Actor() = default;
  virtual ~Actor() = default;

  virtual void Reset(const GameEnv &env) {}

  virtual void ObserveBeforeAct(const GameEnv &env) {}

  virtual void Act(GameEnv &env, int current_player) {}

  virtual void ObserveAfterAct(const GameEnv &env) {}

  virtual void SendExperience(const rela::TensorDict &) {}

  virtual void SetTerminal() {}
};

class BridgeA2CActor : public Actor {
 public:
  BridgeA2CActor(const std::shared_ptr<rela::BatchRunner> &runner,
                 int player_idx)
      : runner_(runner), player_idx_(player_idx) {}

  void ObserveBeforeAct(const GameEnv &env) override;

  void Act(GameEnv &env, int current_player) override;

  void ObserveAfterAct(const GameEnv &env) override {}

  void SendExperience(const rela::TensorDict &) override {}

  void SetTerminal() override {}

 private:
  std::shared_ptr<rela::BatchRunner> runner_;
  rela::Future fut_reply_;
  int player_idx_;
};

class AllPassActor : public Actor {
 public:
  AllPassActor(int player_idx) : player_idx_(player_idx) {}

  void Act(GameEnv &env, int current_player) override {
    if (player_idx_ != current_player) {
      return;
    }
    env.Step(ble::kPass + ble::kBiddingActionBase);
  }

 private:
  int player_idx_;
};

class RandomActor : public Actor {
 public:
  RandomActor(int player_idx) : player_idx_(player_idx) {}

  void ObserveBeforeAct(const GameEnv &env) override {
    if (player_idx_ != env.CurrentPlayer()) {
      return;
    }
    obs_ = env.Feature(-1);
  }

  void Act(GameEnv &env, int current_player) override {
    if (player_idx_ != current_player) {
      return;
    }
    const auto it = obs_.find("legal_move");
    RELA_CHECK(it != obs_.cend(), "Missing key \"legal_move\" in obs.");
    const auto &mov = it->second;
    auto pi = mov / mov.sum(-1, /*keepdim=*/true);
    const auto act = pi.multinomial(1, /*replacement=*/true);
    const int action = act.item<int>();
    env.Step(action);
  }

 private:
  rela::TensorDict obs_;
  int player_idx_;
};

class JPSActor : public Actor {
 public:
  JPSActor(const std::shared_ptr<rela::BatchRunner> &runner, int player_idx)
      : runner_(runner), player_idx_(player_idx) {}

  void ObserveBeforeAct(const GameEnv &env) override;

  void Act(GameEnv &env, int current_player) override;

  void ObserveAfterAct(const GameEnv &env) override {}

 private:
  std::shared_ptr<rela::BatchRunner> runner_;
  rela::FutureReply fut_reply_{};
  int player_idx_;
};

class FFTransitionBuffer {
 public:
  explicit FFTransitionBuffer(float gamma)
      : gamma_(gamma), call_order_(0), terminated_(false) {}

  void PushObs(const rela::TensorDict &obs) {
    RELA_CHECK_EQ(call_order_, 0);
    RELA_CHECK_FALSE(terminated_);
    obs_.push_back(obs);
    ++call_order_;
  }

  void PushAction(const rela::TensorDict &action) {
    RELA_CHECK_EQ(call_order_, 1);
    RELA_CHECK_FALSE(terminated_);
    action_.push_back(action);
    call_order_ = 0;
  }

  void PushReward(const float reward) {
    RELA_CHECK_EQ(call_order_, 0);
    reward_ = reward;
  }

  void PushTerminal() {
    RELA_CHECK_FALSE(terminated_);
    terminated_ = true;
  }

  void Clear() {
    RELA_CHECK(terminated_);
    obs_.clear();
    action_.clear();
    reward_ = std::optional<float>();
    terminated_ = false;
    call_order_ = 0;
  }

  std::vector<rela::FFTransition> PopTransition();

 private:
  float gamma_;

  std::vector<rela::TensorDict> obs_;
  std::vector<rela::TensorDict> action_;
  std::optional<float> reward_;
  bool terminated_;

  int call_order_;
};

class BridgeLSTMActor : public Actor {
 public:
  BridgeLSTMActor(
      const std::shared_ptr<rela::BatchRunner> &runner, int max_len,
      float gamma, std::shared_ptr<rela::RNNPrioritizedReplay> &replay_buffer,
      int player_idx)
      : runner_(runner),
        transition_buffer_(
            std::make_unique<rlcc::RNNTransitionBuffer>(max_len, gamma)),
        replay_buffer_(replay_buffer),
        player_idx_(player_idx),
        table_idx_(0),
        gamma_(gamma) {}

  BridgeLSTMActor(const std::shared_ptr<rela::BatchRunner> &runner,
                  int player_idx)
      : runner_(runner),
        transition_buffer_(nullptr),
        replay_buffer_(nullptr),
        player_idx_(player_idx),
        table_idx_(0),
        gamma_(0) {}

  void Reset(const GameEnv &env) override;

  void ObserveBeforeAct(const GameEnv &env) override;

  void Act(GameEnv &env, int current_player) override;

  void ObserveAfterAct(const GameEnv &env) override;

  void SendExperience(const rela::TensorDict &) override;

  void SetTerminal() override;

 private:
  rela::TensorDict GetH0(std::shared_ptr<rela::BatchRunner> &runner) {
    std::vector<torch::jit::IValue> input{};
    auto model = runner->jitModel();
    auto output = model.get_method("get_h0")(input);
    auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
    return h0;
  }

  static rela::TensorDict SplitPrivatePublic(const rela::TensorDict &feature);

  int player_idx_;
  float gamma_;

  std::shared_ptr<rela::BatchRunner> runner_;
  rela::FutureReply fut_reply_{};
  rela::FutureReply fut_priority0_{};
  // Only used for duplicate envs.
  rela::FutureReply fut_priority1_{};

  rela::TensorDict hidden_;
  rela::TensorDict prev_hidden_;

  // Used for duplicate env.
  std::optional<bool> duplicated{};
  int table_idx_;
  rela::RNNTransition table0_transition_;
  rela::RNNTransition table1_transition_;

  std::unique_ptr<rlcc::RNNTransitionBuffer> transition_buffer_;

  std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer_;
};

class BridgeFFWDActor : public Actor {
 public:
  BridgeFFWDActor(std::shared_ptr<rela::BatchRunner> &runner,
                  float gamma,
                  std::shared_ptr<rela::FFPrioritizedReplay> &replay_buffer,
                  int player_idx)
      : runner_(runner),
        gamma_(gamma),
        transition_buffer_(std::make_unique<FFTransitionBuffer>(gamma)),
        player_idx_(player_idx),
        replay_buffer_(replay_buffer) {
    RELA_CHECK_NOTNULL(replay_buffer_);
  }

  BridgeFFWDActor(std::shared_ptr<rela::BatchRunner> &runner,
                  int player_idx) :
      runner_(runner),
      transition_buffer_(nullptr),
      replay_buffer_(nullptr),
      player_idx_(player_idx) {}
  void Reset(const GameEnv &env) override;
  void ObserveBeforeAct(const GameEnv &env) override;
  void Act(GameEnv &env, int current_player) override;
  void ObserveAfterAct(const GameEnv &env) override;
  void SendExperience(const rela::TensorDict &dict) override;
  void SetTerminal() override;

 private:

  int player_idx_;
  float gamma_;

  std::shared_ptr<rela::BatchRunner> runner_;
  std::shared_ptr<rela::FFPrioritizedReplay> replay_buffer_;
  rela::FutureReply fut_reply_{};
  rela::FutureReply fut_priority0_{};
  // Only used for duplicate envs.
  rela::FutureReply fut_priority1_{};
  std::optional<bool> duplicated{};
  int table_idx_ = 0;
  std::unique_ptr<FFTransitionBuffer> transition_buffer_;
  rela::FFTransition table0_transition_;
  rela::FFTransition table1_transition_;
};

}  // namespace rlcc
#endif  //BRIDGE_ACTOR_H
