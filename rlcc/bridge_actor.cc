#include "bridge_actor.h"
#include <iostream>
#include "bridge_lib/canonical_encoder.h"
#include "rela/utils.h"

namespace rlcc {

void BridgeA2CActor::ObserveBeforeAct(const GameEnv& env) {
  if(env.CurrentPlayer() != player_idx_){
    return;
  }
  torch::NoGradGuard ng;
  rela::TensorDict input;

  input = env.Feature();
  fut_reply_ = runner_->call("act", input);
}

void BridgeA2CActor::Act(GameEnv& env, int current_player) {
  if (current_player != player_idx_) {
    return;
  }
  torch::NoGradGuard ng;
  auto reply = fut_reply_.get();
  int action = reply.at("a").item<int>();
  env.Step(action);
}

void JPSActor::ObserveBeforeAct(const GameEnv& env) {
  if(env.CurrentPlayer() != player_idx_){
    return;
  }
  torch::NoGradGuard ng;
  rela::TensorDict input;

  input = env.Feature();
  fut_reply_ = runner_->call("act_greedy", input);
}

void JPSActor::Act(GameEnv& env, int current_player) {
  if (current_player != player_idx_) {
    return;
  }
  torch::NoGradGuard ng;
  auto reply = fut_reply_.get();
  int action = reply.at("a").item<int>();
  env.Step(action);
}

void AddHid(rela::TensorDict& to, rela::TensorDict& hid) {
  for (auto& kv : hid) {
    auto ret = to.emplace(kv.first, kv.second);
    RELA_CHECK(ret.second);
  }
}

void MoveHid(rela::TensorDict& from, rela::TensorDict& hid) {
  for (auto& kv : hid) {
    auto it = from.find(kv.first);
    RELA_CHECK(it != from.end());
    auto newHid = it->second;
    RELA_CHECK(newHid.sizes() == kv.second.sizes());
    hid[kv.first] = newHid;
    from.erase(it);
  }
}

void BridgePublicLSTMActor::Reset(const GameEnv& env) {
  hidden_ = GetH0(runner_);

  if (transition_buffer_ != nullptr) {
    transition_buffer_->Init(hidden_);
  }
}

void BridgePublicLSTMActor::ObserveBeforeAct(const GameEnv& env) {
  torch::NoGradGuard ng;
  prev_hidden_ = hidden_;

  const auto feature = env.Feature(player_idx_);
  // std::cout << "Get feature\n";
  auto input = SplitPrivatePublic(feature);
  // std::cout << "Split feature\n";

  // Push before we add hidden.
  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushObs(input);
  }

  // Add hidden to input.
  AddHid(input, hidden_);
  // std::cout << "Add hidden to input\n";

  // No-blocking async call to neural network
  fut_reply_ = runner_->call("act", input);
  // std::cout << "call network\n";
}

rela::TensorDict BridgePublicLSTMActor::SplitPrivatePublic(
    const rela::TensorDict& feature) {
  rela::TensorDict res = feature;
  constexpr int kPrivateFeatureSize = ble::kVulnerabilityTensorSize +
                                      ble::kOpeningPassTensorSize +
                                      ble::kBiddingHistoryTensorSize;
  res["publ_s"] =
      feature.at("s").index({torch::indexing::Slice(0, kPrivateFeatureSize)});
  res["priv_s"] = feature.at("s").index(
      {torch::indexing::Slice(kPrivateFeatureSize, ble::kAuctionTensorSize)});
  return res;
}

void BridgePublicLSTMActor::Act(GameEnv& env, int current_player) {
  torch::NoGradGuard ng;

  auto reply = fut_reply_.get();
  // std::cout << "Get reply\n";

  // Update hidden state.
  MoveHid(reply, hidden_);
  // std::cout << "Update hidden\n";

  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushAction(reply);
  }

  if (current_player != player_idx_) {
    return;
  }

  const int action = reply.at("a").item<int>();
  // std::cout << "Get action\n";

  env.Step(action);
}

void BridgePublicLSTMActor::SetTerminal() {
  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushTerminal();
  }
}

void BridgePublicLSTMActor::SendExperience(const rela::TensorDict& t) {
  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushReward(t);
    const auto transition = transition_buffer_->PopTransition();

    replay_buffer_->add(transition);
  }
}

}  // namespace rlcc