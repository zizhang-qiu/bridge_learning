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

  SARTransition(TensorDict& obs, TensorDict& action, torch::Tensor& reward,
                torch::Tensor& terminal)
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

  virtual void Reset(const GameEnv& env) {}

  virtual void ObserveBeforeAct(const GameEnv& env) {}

  virtual void Act(GameEnv& env, int current_player) {}

  virtual void ObserveAfterAct(const GameEnv& env) {}

  virtual void SendExperience(const rela::TensorDict&) {}

  virtual void SetTerminal() {}
};

class BridgeA2CActor : public Actor {
 public:
  BridgeA2CActor(const std::shared_ptr<rela::BatchRunner>& runner,
                 int player_idx)
      : runner_(runner), player_idx_(player_idx) {}

  void ObserveBeforeAct(const GameEnv& env) override;

  void Act(GameEnv& env, int current_player) override;

  void ObserveAfterAct(const GameEnv& env) override {}

  void SendExperience(const rela::TensorDict&) override {}

  void SetTerminal() override {}

 private:
  std::shared_ptr<rela::BatchRunner> runner_;
  rela::Future fut_reply_;
  int player_idx_;
};

class AllPassActor : public Actor {
 public:
  AllPassActor(int player_idx) : player_idx_(player_idx) {}

  void Act(GameEnv& env, int current_player) override {
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

  void ObserveBeforeAct(const GameEnv& env) override {
    if (player_idx_ != env.CurrentPlayer()) {
      return;
    }
    obs_ = env.Feature();
  }

  void Act(GameEnv& env, int current_player) override {
    if (player_idx_ != current_player) {
      return;
    }
    const auto it = obs_.find("legal_move");
    RELA_CHECK(it != obs_.cend(), "Missing key \"legal_move\" in obs.");
    const auto& mov = it->second;
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
  JPSActor(const std::shared_ptr<rela::BatchRunner>& runner, int player_idx)
      : runner_(runner), player_idx_(player_idx) {}

  void ObserveBeforeAct(const GameEnv& env) override;

  void Act(GameEnv& env, int current_player) override;

  void ObserveAfterAct(const GameEnv& env) override {}

 private:
  std::shared_ptr<rela::BatchRunner> runner_;
  rela::FutureReply fut_reply_{};
  int player_idx_;
};

class TransitionBuffer {
 public:
  explicit TransitionBuffer(float gamma)
      : gamma_(gamma), call_order_(0), terminated_(false) {}

  void PushObs(const rela::TensorDict& obs) {
    RELA_CHECK_EQ(call_order_, 0);
    RELA_CHECK_FALSE(terminated_);
    obs_.push_back(obs);
    ++call_order_;
  }

  void PushAction(const rela::TensorDict& action) {
    RELA_CHECK_EQ(call_order_, 1);
    RELA_CHECK_FALSE(terminated_);
    action_.push_back(action);
    ++call_order_;
  }

  void PushReward(const double reward) {
    RELA_CHECK_EQ(call_order_, 2);
    RELA_CHECK_FALSE(terminated_);
    reward_.push_back(reward);
    call_order_ = 0;
  }

  void PushTerminal() {
    RELA_CHECK_FALSE(terminated_);
    terminated_ = true;
  }

  void Clear() {
    CHECK(terminated_);
    obs_.clear();
    action_.clear();
    reward_.clear();
    terminated_ = false;
    call_order_ = 0;
  }

  rela::SARTransition PopTransition(const rela::TensorDict& d = {});

 private:
  float gamma_;

  std::vector<rela::TensorDict> obs_;
  std::vector<rela::TensorDict> action_;
  std::vector<double> reward_;
  bool terminated_;

  int call_order_;
};

class BridgePublicLSTMActor : public Actor {
 public:
  BridgePublicLSTMActor(
      const std::shared_ptr<rela::BatchRunner>& runner, int max_len,
      float gamma, std::shared_ptr<rela::RNNPrioritizedReplay>& replay_buffer,
      int player_idx)
      : runner_(runner),
        transition_buffer_(
            std::make_unique<rlcc::RNNTransitionBuffer>(max_len, gamma)),
        replay_buffer_(replay_buffer),
        player_idx_(player_idx) {}

  BridgePublicLSTMActor(const std::shared_ptr<rela::BatchRunner>& runner,
                        int player_idx)
      : runner_(runner),
        transition_buffer_(nullptr),
        replay_buffer_(nullptr),
        player_idx_(player_idx) {}

  void Reset(const GameEnv& env) override;

  void ObserveBeforeAct(const GameEnv& env) override;

  void Act(GameEnv& env, int current_player) override;

  void ObserveAfterAct(const GameEnv& env) override {}

  void SendExperience(const rela::TensorDict&) override;

  void SetTerminal() override;

 private:
  rela::TensorDict GetH0(std::shared_ptr<rela::BatchRunner>& runner) {
    std::vector<torch::jit::IValue> input{};
    auto model = runner->jitModel();
    auto output = model.get_method("get_h0")(input);
    auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
    return h0;
  }

  rela::TensorDict SplitPrivatePublic(const rela::TensorDict& feature);

  int player_idx_;
  std::shared_ptr<rela::BatchRunner> runner_;
  rela::FutureReply fut_reply_{};

  rela::TensorDict hidden_;
  rela::TensorDict prev_hidden_;

  std::unique_ptr<rlcc::RNNTransitionBuffer> transition_buffer_;

  std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer_;
};

}  // namespace rlcc
#endif  //BRIDGE_ACTOR_H
