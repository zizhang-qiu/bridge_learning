//
// Created by qzz on 2024/1/14.
//

#include "torch_actor.h"

rela::TensorDict TorchActor::GetPolicy(const rela::TensorDict& obs) const {
  torch::NoGradGuard ng;
  auto input_obs = rela::tensor_dict::toTorchDict(obs, model_locker_->device());
  rela::TorchJitInput input;
  input.emplace_back(input_obs);

  int id = -1;
  const auto model = model_locker_->getModel(&id);
  const rela::TorchJitOutput output = model.get_method("get_policy")(input);
  model_locker_->releaseModel(id);
  auto policy = rela::tensor_dict::fromIValue(output, torch::kCPU, /*detach=*/true);
  return policy;
}

rela::TensorDict TorchActor::GetBelief(const rela::TensorDict &obs) const {
  torch::NoGradGuard ng;
  auto input_obs = rela::tensor_dict::toTorchDict(obs, model_locker_->device());
  rela::TorchJitInput input;
  input.emplace_back(input_obs);

  int id = -1;
  const auto model = model_locker_->getModel(&id);
  const rela::TorchJitOutput output = model.get_method("get_belief")(input);
  model_locker_->releaseModel(id);
  auto belief = rela::tensor_dict::fromIValue(output, torch::kCPU, /*detach=*/true);
  return belief;
}
