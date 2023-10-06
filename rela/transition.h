//
// Created by qzz on 2023/9/19.
//

#ifndef BRIDGE_LEARNING_RELA_TRANSITION_H_
#define BRIDGE_LEARNING_RELA_TRANSITION_H_
#include "torch/extension.h"
#include "tensor_dict.h"

namespace rela {

class FFTransition {
 public:
  FFTransition() = default;

  FFTransition(
      TensorDict &obs,
      TensorDict &action,
      torch::Tensor &reward,
      torch::Tensor &terminal,
      torch::Tensor &bootstrap,
      TensorDict &nextObs)
      : obs(obs), action(action), reward(reward), terminal(terminal), bootstrap(bootstrap), nextObs(nextObs) {
  }

  FFTransition index(int i) const;

  FFTransition padLike() const;

  std::vector<torch::jit::IValue> toVectorIValue(const torch::Device &device) const;

  TensorDict toDict();

  TensorDict obs;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  TensorDict nextObs;
};

class RNNTransition {
 public:
  RNNTransition() = default;

  RNNTransition index(int i) const;

  TensorDict toDict();

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;
};

FFTransition makeBatch(
    const std::vector<FFTransition> &transitions, const std::string &device);

RNNTransition makeBatch(
    const std::vector<RNNTransition> &transitions, const std::string &device);

TensorDict makeBatch(
    const std::vector<TensorDict> &transitions, const std::string &device);

}
#endif //BRIDGE_LEARNING_RELA_TRANSITION_H_