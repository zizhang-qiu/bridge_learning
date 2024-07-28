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
    if (env_->Terminated()) {
      env_->Reset();
    }
    Reset();
  }

  void Reset() override{
    for (int i = 0; i < actors_.size(); ++i){
      actors_[i] -> Reset(*env_);
    }
  }

  void ObserveBeforeAct() override {

    // const int current_player = env_->CurrentPlayer();

    // actors_[current_player]->ObserveBeforeAct(*env_);
    for(size_t i=0; i<actors_.size(); ++i){
      actors_[i]->ObserveBeforeAct(*env_);
    }

  }

  void Act() override {
    // // std::cout << "Enter act" << std::endl;
    const int current_player = env_->CurrentPlayer();
    // // std::cout << "current player: " << current_player << std::endl;
    // actors_[current_player]->Act(*env_);
    // // std::cout << "After actors_[current_player]->Act(*env_)\n";

    for(size_t i=0; i<actors_.size(); ++i){
      actors_[i]->Act(*env_, current_player);
    }

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
      rela::TensorDict d = {{"r", torch::tensor(last_rewards_[i])}};
      actors_[i]->SendExperience(d);
    }
  }

  void PostSendExperience() override {
    if (!env_->Terminated()) {
      return;
    }
    env_->Reset();
    Reset();
  }

  const std::shared_ptr<GameEnv>& GetEnv() const override { return env_; }

 private:
  std::vector<float> last_rewards_;
  std::shared_ptr<GameEnv> env_;
};
}  // namespace rlcc

#endif /* RLCC_BRIDGE_ENV_ACTOR_H */
