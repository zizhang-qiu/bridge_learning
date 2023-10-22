//
// Created by qzz on 2023/9/19.
//
#include "transition.h"
#include "utils.h"

namespace rela{
FFTransition FFTransition::index(int i) const {
  FFTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.terminal = terminal[i];
  element.bootstrap = bootstrap[i];

  for (auto& name2tensor : nextObs) {
    element.nextObs.insert({name2tensor.first, name2tensor.second[i]});
  }

  return element;
}

FFTransition FFTransition::padLike() const {
  FFTransition pad;

  pad.obs = tensor_dict::zerosLike(obs);
  pad.action = tensor_dict::zerosLike(action);
  pad.reward = torch::zeros_like(reward);
  pad.terminal = torch::ones_like(terminal);
  pad.bootstrap = torch::zeros_like(bootstrap);
  pad.nextObs = tensor_dict::zerosLike(nextObs);

  return pad;
}

std::vector<torch::jit::IValue> FFTransition::toVectorIValue(
    const torch::Device& device) const {
  std::vector<torch::jit::IValue> vec;
  vec.push_back(tensor_dict::toIValue(obs, device));
  vec.push_back(tensor_dict::toIValue(action, device));
  vec.emplace_back(reward.to(device));
  vec.emplace_back(terminal.to(device));
  vec.emplace_back(bootstrap.to(device));
  vec.push_back(tensor_dict::toIValue(nextObs, device));
  return vec;
}

TensorDict FFTransition::toDict() {
  auto dict = obs;
  for (auto& kv : nextObs) {
    dict["next_" + kv.first] = kv.second;
  }

  for (auto& kv : action) {
    auto ret = dict.emplace(kv.first, kv.second);
    assert(ret.second);
  }

  auto ret = dict.emplace("reward", reward);
  assert(ret.second);
  ret = dict.emplace("terminal", terminal);
  assert(ret.second);
  ret = dict.emplace("bootstrap", bootstrap);
  assert(ret.second);
  return dict;
}

RNNTransition RNNTransition::index(int i) const {
  RNNTransition element;

  for (auto& name2tensor : obs) {
    element.obs.insert({name2tensor.first, name2tensor.second[i]});
  }
  for (auto& name2tensor : h0) {
    auto t = name2tensor.second.narrow(1, i, 1).squeeze(1);
    element.h0.insert({name2tensor.first, t});
  }
  for (auto& name2tensor : action) {
    element.action.insert({name2tensor.first, name2tensor.second[i]});
  }

  element.reward = reward[i];
  element.terminal = terminal[i];
  element.bootstrap = bootstrap[i];
  element.seqLen = seqLen[i];
  return element;
}

TensorDict RNNTransition::toDict() {
  auto dict = obs;

  for (auto& kv : action) {
    auto ret = dict.emplace(kv.first, kv.second);
    assert(ret.second);
  }

  for (auto& kv : h0) {
    auto ret = dict.emplace(kv.first, kv.second);
    assert(ret.second);
  }

  auto ret = dict.emplace("reward", reward);
  assert(ret.second);
  ret = dict.emplace("terminal", terminal);
  assert(ret.second);
  ret = dict.emplace("bootstrap", bootstrap);
  assert(ret.second);
  ret = dict.emplace("seq_len", seqLen);
  assert(ret.second);
  return dict;
}

RNNTransition makeBatch(
    const std::vector<RNNTransition>& transitions, const std::string& device) {
  std::vector<TensorDict> obsVec;
  std::vector<TensorDict> h0Vec;
  std::vector<TensorDict> actionVec;
  std::vector<torch::Tensor> rewardVec;
  std::vector<torch::Tensor> terminalVec;
  std::vector<torch::Tensor> bootstrapVec;
  std::vector<torch::Tensor> seqLenVec;

  for (const auto & transition : transitions) {
    obsVec.push_back(transition.obs);
    h0Vec.push_back(transition.h0);
    actionVec.push_back(transition.action);
    rewardVec.push_back(transition.reward);
    terminalVec.push_back(transition.terminal);
    bootstrapVec.push_back(transition.bootstrap);
    seqLenVec.push_back(transition.seqLen);
  }

  RNNTransition batch;
  batch.obs = tensor_dict::stack(obsVec, 1);
  batch.h0 = tensor_dict::stack(h0Vec, 1);  // 1 is batch for rnn hid
  batch.action = tensor_dict::stack(actionVec, 1);
  batch.reward = torch::stack(rewardVec, 1);
  batch.terminal = torch::stack(terminalVec, 1);
  batch.bootstrap = torch::stack(bootstrapVec, 1);
  batch.seqLen = torch::stack(seqLenVec, 0);

  if (device != "cpu") {
    auto d = torch::Device(device);
    auto toDevice = [&](const torch::Tensor& t) { return t.to(d); };
    batch.obs = tensor_dict::apply(batch.obs, toDevice);
    batch.h0 = tensor_dict::apply(batch.h0, toDevice);
    batch.action = tensor_dict::apply(batch.action, toDevice);
    batch.reward = batch.reward.to(d);
    batch.terminal = batch.terminal.to(d);
    batch.bootstrap = batch.bootstrap.to(d);
    batch.seqLen = batch.seqLen.to(d);
  }

  return batch;
}

TensorDict makeBatch(
    const std::vector<TensorDict>& transitions, const std::string& device) {
  auto batch = tensor_dict::stack(transitions, 0);
  if (device != "cpu") {
    auto d = torch::Device(device);
    for (auto& kv : batch) {
      batch[kv.first] = kv.second.to(d);
    }
  }
  return batch;
}
}