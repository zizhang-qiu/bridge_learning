#ifndef RLCC_BRIDGE_ENV_ACTOR_H
#define RLCC_BRIDGE_ENV_ACTOR_H

#include <memory>
#include <vector>
#include "env_actor.h"
#include "rela/types.h"
#include "rlcc/env.h"

namespace rlcc {
class BridgeEnvActor : public EnvActor {
 public:
  BridgeEnvActor(const std::shared_ptr<GameEnv>& env,
                 const EnvActorOptions& options,
                 std::vector<std::shared_ptr<Actor>> actors)
      : EnvActor(actors, options), env_(env) {
    CheckValid(*env);
  }

  void ObserveBeforeAct() override {
    // std::cout << "Enter observe before act" << std::endl;
    const int current_player = env_->CurrentPlayer();

    // const rela::TensorDict obs = env_->Feature();

    actors_[current_player]->ObserveBeforeAct(*env_);
    // std::cout << "Leave observe before act" << std::endl;
  }

  void Act() override {
    // std::cout << "Enter act" << std::endl;
    const int current_player = env_->CurrentPlayer();
    // std::cout << "current player: " << current_player << std::endl;
    actors_[current_player]->Act(*env_);

    bool terminated = env_->Terminated();
    if (terminated) {
      for (int i = 0; i < actors_.size(); ++i) {
        actors_[i]->SetTerminal();
      }
      const auto rewards = env_->Rewards();
      last_rewards_ = rewards;
      EnvTerminate(&rewards, env_->ToString());
    }
    // std::cout << "Leave act" << std::endl;
  }

  void ObserveAfterAct() override {}

  void SendExperience() override {
    if (!env_->Terminated()) {
      return;
    }
    const auto rewards = env_->Rewards();

    for (int i = 0; i < actors_.size(); ++i) {
      rela::TensorDict d = {{"reward", torch::tensor(last_rewards_[i])}};
      actors_[i]->SendExperience(d);
    }
  }

  void PostSendExperience() override {
    if (!env_->Terminated()) {
      return;
    }
    env_->Reset();
  }

  const std::shared_ptr<GameEnv>& GetEnv() const override{
    return env_;
  }

 private:
  std::vector<float> last_rewards_;
  std::shared_ptr<GameEnv> env_;
};
}  // namespace rlcc

#endif /* RLCC_BRIDGE_ENV_ACTOR_H */
