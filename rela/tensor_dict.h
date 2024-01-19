//
// Created by qzz on 2023/9/19.
//

#ifndef BRIDGE_LEARNING_RELA_TENSOR_DICT_H_
#define BRIDGE_LEARNING_RELA_TENSOR_DICT_H_
#include "torch/extension.h"
#include "torch/torch.h"
#include <unordered_map>
#include <string>

#include "types.h"

namespace rela {
namespace tensor_dict {
inline void compareShape(const TensorDict& src, const TensorDict& dest) {
  if (src.size() != dest.size()) {
    std::cout << "src.size()[" << src.size() << "] != dest.size()[" << dest.size() << "]"
        << std::endl;
    std::cout << "src keys: ";
    for (const auto& p : src)
      std::cout << p.first << " ";
    std::cout << "dest keys: ";
    for (const auto& p : dest)
      std::cout << p.first << " ";
    std::cout << std::endl;
    assert(false);
  }

  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    const auto& destTensor = dest.at(name);
    if (destTensor.sizes() != srcTensor.sizes()) {
      std::cout << name << ", dstSize: " << destTensor.sizes()
          << ", srcSize: " << srcTensor.sizes() << std::endl;
      assert(false);
    }
  }
}

inline void copy(const TensorDict& src, TensorDict& dest) {
  compareShape(src, dest);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    destTensor.copy_(srcTensor);
  }
}

inline void copy(const TensorDict& src, TensorDict& dest, const torch::Tensor& index) {
  assert(src.size() == dest.size());
  assert(index.size(0) > 0);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    assert(destTensor.dtype() == srcTensor.dtype());
    assert(index.size(0) == srcTensor.size(0));
    destTensor.index_copy_(0, index, srcTensor);
  }
}

inline bool eq(const TensorDict& d0, const TensorDict& d1) {
  if (d0.size() != d1.size()) {
    return false;
  }

  for (const auto& name2tensor : d0) {
    auto key = name2tensor.first;
    if ((d1.at(key) != name2tensor.second).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

/*
 * indexes into a TensorDict
 */
inline TensorDict index(const TensorDict& batch, size_t i) {
  TensorDict result;
  for (const auto& name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second[i]});
  }
  return result;
}

inline TensorDict narrow(
  const TensorDict& batch,
  size_t dim,
  size_t i,
  size_t len,
  bool squeeze) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    auto t = name2tensor.second.narrow(dim, i, len);
    if (squeeze) {
      assert(len == 1);
      t = t.squeeze(dim);
    }
    result.insert({name2tensor.first, std::move(t)});
  }
  return result;
}

inline TensorDict clone(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, name2tensor.second.clone()});
  }
  return output;
}

inline TensorDict zerosLike(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, torch::zeros_like(name2tensor.second)});
  }
  return output;
}

// TODO: rewrite the above functions with this template
template<typename Func>
inline TensorDict apply(TensorDict& dict, Func f) {
  TensorDict output;
  for (const auto& name2tensor : dict) {
    auto tensor = f(name2tensor.second);
    output.insert({name2tensor.first, tensor});
  }
  return output;
}

inline TensorDict stack(const std::vector<TensorDict>& vec, int stack_dim) {
  assert(vec.size() >= 1);
  size_t numKey = vec[0].size();
  TensorDict ret;
  for (auto& name2tensor : vec[0]) {
    std::vector<torch::Tensor> buffer(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
      if (vec[i].size() != numKey) {
        std::cout << "i: " << i << std::endl;
        std::cout << "ref keys: " << std::endl;
        for (auto& kv : vec[0]) {
          std::cout << kv.first << ", ";
        }
        std::cout << std::endl;

        std::cout << "new keys: " << std::endl;
        for (auto& kv : vec[i]) {
          std::cout << kv.first << ", ";
        }
        std::cout << std::endl;
      }
      assert(vec[i].size() == numKey);
      buffer[i] = vec[i].at(name2tensor.first);
    }
    ret[name2tensor.first] = torch::stack(buffer, stack_dim);
  }
  return ret;
}

inline TensorDict fromIValue(
  const torch::jit::IValue& value,
  torch::DeviceType device,
  bool detach) {
  std::unordered_map<std::string, torch::Tensor> map;
  auto dict = value.toGenericDict();
  for (auto& name2tensor : dict) {
    auto name = name2tensor.key().toString();
    torch::Tensor tensor = name2tensor.value().toTensor();
    tensor = tensor.to(device);
    if (detach) {
      tensor = tensor.detach();
    }
    map.insert({name->string(), tensor});
  }
  return map;
}

// TODO: this may be simplified with constructor in the future version
inline torch::jit::IValue toIValue(
  const TensorDict& tensorDict,
  const torch::Device& device) {
  torch::Dict<std::string, torch::Tensor> dict;
  for (const auto& name2tensor : tensorDict) {
    dict.insert(name2tensor.first, name2tensor.second.to(device));
  }
  return {dict};
}

inline std::vector<std::string> getKeys(const TensorDict& d) {
  std::vector<std::string> keys;
  for (auto& kv : d) {
    keys.push_back(kv.first);
  }
  return keys;
}

inline TorchTensorDict toTorchDict(const TensorDict& tensorDict,
                                   const torch::Device& device) {
  TorchTensorDict dict;
  for (const auto& name2tensor : tensorDict) {
    dict.insert(name2tensor.first, name2tensor.second.to(device));
  }
  return dict;
}

inline void print(const TensorDict& tensorDict) {
  for (const auto& kv : tensorDict) {
    std::cout << kv.first << ":\n" << kv.second << std::endl;
  }
}

inline TensorDict join(const TensorVecDict& dict, int64_t dim) {
  TensorDict result;
  for (const auto& it : dict) {
    result.emplace(it.first, torch::stack(it.second, dim));
  }
  return result;
}

inline void tensorVecDictAppend(TensorVecDict& batch, const TensorDict& input) {
  for (auto& name2tensor : input) {
    auto it = batch.find(name2tensor.first);
    if (it == batch.end()) {
      std::vector<torch::Tensor> singleton = {name2tensor.second};
      batch.insert({name2tensor.first, singleton});
    }
    else {
      it->second.push_back(name2tensor.second);
    }
  }
}

inline void assertKeyExists(const TensorDict& tensorDict,
                            const std::vector<std::string>& keys) {
  for (const auto& k : keys) {
    if (tensorDict.find(k) == tensorDict.end()) {
      std::cout << "Key " << k << " does not exist! " << std::endl;
      std::cout << "Checking keys: " << std::endl;
      for (const auto& kk : keys) {
        std::cout << kk << ", ";
      }
      std::cout << std::endl;
      assert(false);
    }
  }
}

inline void _combineTensorDictArgs(TensorDict& combined,
                                   int i,
                                   const TensorDict& d) {
  for (const auto& kv : d) {
    combined[std::to_string(i) + "." + kv.first] = kv.second;
  }
}

inline TensorDict combineTensorDictArgs(const TensorDict& d1,
                                        const TensorDict& d2) {
  TensorDict res;
  _combineTensorDictArgs(res, 0, d1);
  _combineTensorDictArgs(res, 1, d2);
  return res;
}

inline TensorDict toDevice(const TensorDict& dict,
  const torch::Device& device) {
  TensorDict res{};
  for (const auto &kv : dict) {
    res[kv.first] = kv.second.to(device);
  }
  return res;
}

inline TensorDict toDevice(const TensorDict& dict,
  const std::string& device) {
  const auto d = torch::Device{device};
  return toDevice(dict, d);
}


}
}

#endif //BRIDGE_LEARNING_RELA_TENSOR_DICT_H_
