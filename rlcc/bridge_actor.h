//
// Created by qzz on 2024/1/23.
//

#ifndef BRIDGE_ACTOR_H
#define BRIDGE_ACTOR_H
#include <memory>
#include "bridge_lib/bridge_utils.h"
#include "rela/batch_runner.h"
#include "rela/batcher.h"
#include "rela/tensor_dict.h"
#include "rela/types.h"
#include "rlcc/env.h"

namespace rlcc {

namespace ble = bridge_learning_env;

class Actor {
 public:
  Actor() = default;
  virtual ~Actor() = default;

  virtual void ObserveBeforeAct(const GameEnv& env) {}

  virtual void Act(GameEnv& env) {}

  virtual void ObserveAfterAct(const GameEnv& env) {}

  virtual void SendExperience(const rela::TensorDict&) {}

  virtual void SetTerminal() {}
};

class BridgeA2CActor : public Actor {
 public:
  BridgeA2CActor(const std::shared_ptr<rela::BatchRunner>& runner)
      : runner_(runner) {}

  void ObserveBeforeAct(const GameEnv& env) override;

  void Act(GameEnv& env) override;

  void ObserveAfterAct(const GameEnv& env) override {}

  void SendExperience(const rela::TensorDict&) override {}

  void SetTerminal() override {}

 private:
  std::shared_ptr<rela::BatchRunner> runner_;
  rela::Future fut_reply_;
};

class AllPassActor : public Actor {
 public:
  AllPassActor() = default;

  void Act(GameEnv& env) { env.Step(ble::kPass + ble::kBiddingActionBase); }
};

class JPSActor : public Actor {
 public:
  JPSActor(const std::shared_ptr<rela::BatchRunner>& runner)
      : runner_(runner) {}

  void ObserveBeforeAct(const GameEnv& env) override;

  void Act(GameEnv& env) override;

  void ObserveAfterAct(const GameEnv& env) override {}

 private:
  std::shared_ptr<rela::BatchRunner> runner_;
  rela::FutureReply fut_reply_{};
};
}  // namespace rlcc
#endif  //BRIDGE_ACTOR_H
