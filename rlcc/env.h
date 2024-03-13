#ifndef RLCC_ENV_H
#define RLCC_ENV_H

#include <ostream>
#include <tuple>
#include "rela/types.h"
#include "torch/torch.h"

#include "rela/tensor_dict.h"

namespace rlcc {

struct EnvSpec {
  int num_players = -1;
  int num_partnerships = -1;
};

// An environment only supports reward at terminal (game-style)
class GameEnv {
 public:
  GameEnv() = default;

  GameEnv(const GameEnv&) = default;

  virtual ~GameEnv() = default;
  // State retrievers.
  virtual bool Terminated() const = 0;

  virtual int MaxNumAction() const = 0;

  virtual void Step(int action) = 0;

  // Return false if the environment won't work anymore (e.g., it has gone
  // through all samples).
  virtual bool Reset() = 0;

  virtual void SetReply(const rela::TensorDict&) {}

  virtual std::vector<int> LegalActions() const {
    // Get legal actions for that particular state.
    // Default behavior: everything is legal. The derived class can override
    // this.
    if (Terminated())
      return {};
    std::vector<int> actions(MaxNumAction());
    for (int i = 0; i < (int)actions.size(); ++i) {
      actions[i] = i;
    }
    return actions;
  }

  virtual int CurrentPlayer() const = 0;

  virtual int CurrentPartnership() const = 0;

  virtual float PlayerReward(int player) const = 0;

  // Rewards for all players.
  virtual std::vector<float> Rewards() const = 0;

  virtual std::string ToString() const = 0;

  virtual EnvSpec Spec() const = 0;

  virtual rela::TensorDict Feature() const = 0;

 private:
  torch::Tensor LegalActionMask() const {
    torch::Tensor legal_move = torch::zeros({MaxNumAction()});
    auto legals = LegalActions();

    auto f = legal_move.accessor<float, 1>();
    for (const auto& idx : legals) {
      f[idx] = 1.0;
    }
    return legal_move;
  }
};

// An environment supports reward after each action (RL-style)
class RLEnv {
 public:
  RLEnv() = default;
  virtual ~RLEnv() = default;

  // State retrievers.
  virtual bool Terminated() const = 0;

  virtual int MaxNumAction() const = 0;

  virtual std::tuple<rela::TensorDict, float, bool> Step(int action) = 0;

  // Return false if the environment won't work anymore (e.g., it has gone
  // through all samples).
  virtual bool Reset() = 0;

  virtual void SetReply(const rela::TensorDict&) {}

  virtual std::vector<int> LegalActions() const {
    // Get legal actions for that particular state.
    // Default behavior: everything is legal. The derived class can override
    // this.
    if (Terminated())
      return {};
    std::vector<int> actions(MaxNumAction());
    for (int i = 0; i < (int)actions.size(); ++i) {
      actions[i] = i;
    }
    return actions;
  }

  virtual int CurrentPlayer() const = 0;

  virtual int CurrentPartnership() const = 0;

  virtual float PlayerReward(int player) const = 0;

  // Rewards for all players.
  virtual std::vector<float> Rewards() const = 0;
};

}  // namespace rlcc

#endif /* RLCC_ENV_H */
