#include "bridge_actor.h"
#include <iostream>
#include "rela/utils.h"

namespace rlcc {

void BridgeA2CActor::ObserveBeforeAct(const GameEnv& env) {
  torch::NoGradGuard ng;
  rela::TensorDict input;

  input = env.Feature();
  fut_reply_ = runner_->call("act", input);
}

void BridgeA2CActor::Act(GameEnv& env) {
  torch::NoGradGuard ng;
  auto reply = fut_reply_.get();
  int action = reply.at("a").item<int>();
  env.Step(action);
}

void JPSActor::ObserveBeforeAct(const GameEnv& env) {
  torch::NoGradGuard ng;
  rela::TensorDict input;

  input = env.Feature();
  // rela::utils::printMapKey(input);
  fut_reply_ = runner_->call("act_greedy", input);
  // std::cout << "Get future reply" << std::endl;
}

void JPSActor::Act(GameEnv& env) {
  // std::cout << "Enter BaselineActor::Act" << std::endl;
  torch::NoGradGuard ng;
  // std::cout << fut_reply_.isNull() << std::endl;
  // std::cout << "Env:\n" << env.ToString() << std::endl;
  auto reply = fut_reply_.get();
  // std::cout << "reply" << std::endl;
  // rela::utils::printMap(reply);
  int action = reply.at("a").item<int>();
  // std::cout << "action:" << action << std::endl;
  env.Step(action);
}
}  // namespace rlcc