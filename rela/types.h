//
// Created by qzz on 2024/1/14.
//

#ifndef BRIDGE_LEARNING_RELA_TYPES_H_
#define BRIDGE_LEARNING_RELA_TYPES_H_
#include <string>
#include <unordered_map>

#include "torch/torch.h"

namespace rela {
using TensorDict = std::unordered_map<std::string, torch::Tensor>;
using TensorVecDict =
std::unordered_map<std::string, std::vector<torch::Tensor>>;

using TorchTensorDict = torch::Dict<std::string, torch::Tensor>;
using TorchJitInput = std::vector<torch::jit::IValue>;
using TorchJitOutput = torch::jit::IValue;
using TorchJitModel = torch::jit::script::Module;
}
#endif //BRIDGE_LEARNING_RELA_TYPES_H_
