//
// Created by qzz on 2024/1/14.
//

#include "torch_actor.h"

rela::TensorDict TorchActor::GetPolicy(const rela::TensorDict& obs){
  torch::NoGradGuard ng;
  fut_policy_ = runner_->call("get_policy", obs);
  auto policy = fut_policy_.get();
  return policy;
}

rela::TensorDict TorchActor::GetBelief(const rela::TensorDict &obs){
  torch::NoGradGuard ng;
  fut_belief_ = runner_->call("get_belief", obs);
  auto belief = fut_belief_.get();
  return belief;
}
