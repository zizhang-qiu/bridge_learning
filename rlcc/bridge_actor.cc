#include "bridge_actor.h"
#include <iostream>
#include "bridge_lib/canonical_encoder.h"
#include "rela/utils.h"

namespace rlcc {

void BridgeA2CActor::ObserveBeforeAct(const GameEnv &env) {
  if (env.CurrentPlayer() != player_idx_) {
    return;
  }
  torch::NoGradGuard ng;

  const rela::TensorDict input = env.Feature(player_idx_);
  fut_reply_ = runner_->call("act", input);
}

void BridgeA2CActor::Act(GameEnv &env, int current_player) {
  if (current_player != player_idx_) {
    return;
  }
  torch::NoGradGuard ng;
  const auto reply = fut_reply_.get();
  int action = reply.at("a").item<int>();
  env.Step(action);
}

void JPSActor::ObserveBeforeAct(const GameEnv &env) {
  if (env.CurrentPlayer() != player_idx_) {
    return;
  }
  torch::NoGradGuard ng;

  const rela::TensorDict input = env.Feature(player_idx_);
  fut_reply_ = runner_->call("act_greedy", input);
}

void JPSActor::Act(GameEnv &env, int current_player) {
  if (current_player != player_idx_) {
    return;
  }
  torch::NoGradGuard ng;
  auto reply = fut_reply_.get();
  int action = reply.at("a").item<int>();
  env.Step(action);
}

std::vector<rela::FFTransition> FFTransitionBuffer::PopTransition() {
  RELA_CHECK(terminated_);
  RELA_CHECK(reward_.has_value());
  RELA_CHECK_EQ(obs_.size(), action_.size());

//  std::cout << "reward_: " << reward_.value() << std::endl;
  std::vector<float> rewards(obs_.size());
  rewards.back() = reward_.value();
  if (rewards.size() > 1) {
    for (int i = static_cast<int>(rewards.size()) - 2; i >= 0; --i) {
      rewards[i] += gamma_ * rewards[i + 1];
    }
  }
//  std::cout << "rewards: \n";
//  rela::utils::printVector(rewards);

//  std::cout << "obs size: " << obs_.size() << std::endl;
  std::vector<rela::FFTransition> transitions;
  transitions.reserve(obs_.size());
  for (size_t i = 0; i < obs_.size(); ++i) {
    torch::Tensor reward = torch::tensor(rewards[i], {torch::kFloat32});
    torch::Tensor terminal = torch::tensor(i == (obs_.size() - 1), {torch::kFloat32});
    torch::Tensor bootstrap = torch::tensor(1, {torch::kFloat32});
    rela::TensorDict next_obs = i == (obs_.size() - 1) ? rela::tensor_dict::zerosLike(obs_[i]) : obs_[i + 1];
//    rela::FFTransition transition{
//        /*obs=*/obs_[i],
//        /*action=*/action_[i],
//        /*reward=*/reward,
//        /*terminal=*/terminal,
//        /*bootstrap=*/bootstrap,
//        /*nextObs=*/next_obs
//    };
    transitions.emplace_back(/*obs=*/obs_[i],
        /*action=*/action_[i],
        /*reward=*/reward,
        /*terminal=*/terminal,
        /*bootstrap=*/bootstrap,
        /*nextObs=*/next_obs);
  }
  Clear();

  return transitions;
}

void AddHid(rela::TensorDict &to, rela::TensorDict &hid) {
  for (auto &kv : hid) {
    auto ret = to.emplace(kv.first, kv.second);
    RELA_CHECK(ret.second);
  }
}

void MoveHid(rela::TensorDict &from, rela::TensorDict &hid) {
  for (auto &kv : hid) {
    auto it = from.find(kv.first);
    RELA_CHECK(it != from.end());
    auto newHid = it->second;
    RELA_CHECK(newHid.sizes() == kv.second.sizes());
    hid[kv.first] = newHid;
    from.erase(it);
  }
}

bool IsEnvDuplicated(const rela::TensorDict &feature) {
  return feature.count("table_idx");
}

void BridgeLSTMActor::Reset(const GameEnv &env) {
  hidden_ = GetH0(runner_);
  // Check if the env is duplicated.
  if (!duplicated.has_value()) {
    const auto feature = env.Feature(player_idx_);
    duplicated = IsEnvDuplicated(feature);
  }

  if (transition_buffer_ != nullptr) {
    transition_buffer_->Init(hidden_);
  }
}

void BridgeLSTMActor::ObserveBeforeAct(const GameEnv &env) {
  torch::NoGradGuard ng;
  prev_hidden_ = hidden_;

  const auto feature = env.Feature(player_idx_);
  auto input = SplitPrivatePublic(feature);


  // Push before we add hidden.
  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushObs(input);
  }

  // Add hidden to input.
  AddHid(input, hidden_);
//  for (const auto &kv : input) {
//    std::cout << kv.first << ": " << kv.second.sizes() << std::endl;
//  }


  // No-blocking async call to neural network
  fut_reply_ = runner_->call("act", input);

}

rela::TensorDict BridgeLSTMActor::SplitPrivatePublic(
    const rela::TensorDict &feature) {
  rela::TensorDict res = {};
  const int kPrivateFeatureSize = static_cast<int>(feature.at("s").size(0)) - ble::kNumCards;
  res["publ_s"] =
      feature.at("s").index({torch::indexing::Slice(0, kPrivateFeatureSize)});
  res["priv_s"] = feature.at("s");
  res["legal_move"] = feature.at("legal_move");
  return res;
}

void BridgeLSTMActor::Act(GameEnv &env, int current_player) {
//  std::cout << "Enter act." << std::endl;
  torch::NoGradGuard ng;

  auto reply = fut_reply_.get();
//  std::cout << "get reply." << std::endl;

  // Update hidden state.
  MoveHid(reply, hidden_);

  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushAction(reply);
  }

  if (current_player != player_idx_) {
    return;
  }

  const int action = reply.at("a").item<int>();

  env.Step(action);
}

void UpdateRNNTransitionReward(rela::RNNTransition &transition,
                               const float reward,
                               const float gamma) {
  const int len = transition.seqLen.item<int>();
  transition.reward[len - 1] = reward;
  for (int i = len - 2; i >= 0; --i) {
    transition.reward[i] += transition.reward[i + 1] * gamma;
  }
}

void BridgeLSTMActor::ObserveAfterAct(const GameEnv &env) {
  if (!duplicated.value()) {
    return;
  }

  if (env.Terminated()) {
    table_idx_ = 0;
    return;
  }

  const auto feature = env.Feature(player_idx_);
  // If the first table of duplicate environment is terminated, we have to pop the transition first.
  if (table_idx_ == 0 && feature.at("table_idx").item<int>() == 1) {
    table_idx_ = 1;
    // Start a new transition.
    hidden_ = GetH0(runner_);
    if (transition_buffer_ != nullptr) {
      // Fake reward here. We can update it later because of duplicate setting.
      transition_buffer_->PushTerminal();
      transition_buffer_->PushReward({{"r", torch::tensor(0.)}});
      table0_transition_ = transition_buffer_->PopTransition();
      // Start a new transition.
      transition_buffer_->Init(hidden_);
    }
  }

}

void BridgeLSTMActor::SetTerminal() {
  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushTerminal();
  }
}

void BridgeLSTMActor::SendExperience(const rela::TensorDict &t) {
  torch::NoGradGuard ng;
  if (replay_buffer_ == nullptr) {
    return;
  }

  // Deal with stored priority and transition.
  if (!fut_priority0_.isNull()) {
    auto priority = fut_priority0_.get()["priority"].item<float>();
//    std::cout << "transition 0 priority: " << priority << std::endl;
    replay_buffer_->add(table0_transition_, priority);
  }

  if (!fut_priority1_.isNull()) {
    auto priority = fut_priority1_.get()["priority"].item<float>();
//    std::cout << "transition 1 priority: " << priority << std::endl;
    replay_buffer_->add(table1_transition_, priority);
  }

  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushReward(t);
//    std::cout << "duplicate.value()=" << duplicated.value() << std::endl;
    if (duplicated.value()) {
      table1_transition_ = transition_buffer_->PopTransition();
      UpdateRNNTransitionReward(table0_transition_, t.at("r").item<float>(), gamma_);
//      float priority = std::abs(t.at("r").item<float>());
//      float priority = 1.0;
      rela::TensorDict input0 = table0_transition_.toDict();
      rela::TensorDict input1 = table1_transition_.toDict();
      fut_priority0_ = runner_->call("compute_priority", input0);
      fut_priority1_ = runner_->call("compute_priority", input1);
//      replay_buffer_->add(table0_transition_, priority);
//      replay_buffer_->add(table1_transition, priority);
    } else {
      table0_transition_ = transition_buffer_->PopTransition();
      rela::TensorDict input0 = table0_transition_.toDict();
      fut_priority0_ = runner_->call("compute_priority", input0);
//      replay_buffer_->add(transition);
    }
  }
}

void BridgeFFWDActor::Reset(const GameEnv &env) {
  // Check if the env is duplicated.
  if (!duplicated.has_value()) {
    const auto feature = env.Feature(player_idx_);
    duplicated = IsEnvDuplicated(feature);
  }
}

void BridgeFFWDActor::ObserveBeforeAct(const GameEnv &env) {
  if (env.CurrentPlayer() != player_idx_) {
    // Do not call network if it's not player's turn for ffwd actor.
    return;
  }
  torch::NoGradGuard ng;
  auto feature = env.Feature(player_idx_);
  const int perf_size = static_cast<int>(feature.at("s").size(0));
  const int priv_size = perf_size - (ble::kNumPlayers - 1) * ble::kNumCards;
  const int publ_size = perf_size - (ble::kNumPlayers) * ble::kNumCards;
  feature["perf_s"] = feature.at("s");
  feature["publ_s"] = feature.at("s").index(
      {torch::indexing::Slice(0, publ_size)});
  feature["priv_s"] = feature.at("s").index(
      {torch::indexing::Slice(0, priv_size)});
  feature.erase("s");
  feature["legal_move"] = feature.at("legal_move").index(
      {torch::indexing::Slice(ble::kNumCards, -1)});
  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushObs(feature);
  }

  // No-blocking async call to neural network
  fut_reply_ = runner_->call("act", feature);
}

void BridgeFFWDActor::Act(GameEnv &env, int current_player) {
  if (current_player != player_idx_) {
    return;
  }
  torch::NoGradGuard ng;
  auto reply = fut_reply_.get();

  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushAction(reply);
  }

  const int action = reply.at("a").item<int>();

  env.Step(action);
}
void BridgeFFWDActor::ObserveAfterAct(const GameEnv &env) {
  if (!duplicated.value()) {
    return;
  }

  if (env.Terminated()) {
    table_idx_ = 0;
    return;
  }

  const auto feature = env.Feature(player_idx_);
  // If the first table of duplicate environment is terminated, we have to pop the transition first.
  if (table_idx_ == 0 && feature.at("table_idx").item<int>() == 1) {
    table_idx_ = 1;
    if (transition_buffer_ != nullptr) {
      transition_buffer_->PushTerminal();
      transition_buffer_->PushReward(0.0); // Fake reward here.
      const auto transitions = transition_buffer_->PopTransition();
      table0_transition_ = rela::makeBatch(transitions, "cpu");
    }
  }
}
void BridgeFFWDActor::SendExperience(const rela::TensorDict &dict) {
//  std::cout << "Actor " << player_idx_ << "Send Experience." << std::endl;
//  std::cout << "replay_buffer_ == nullptr?" << (replay_buffer_ == nullptr) << std::endl;
  torch::NoGradGuard ng;
  if (replay_buffer_ == nullptr) {
    return;
  }
  transition_buffer_->PushReward(dict.at("r").item<float>());
  if (duplicated.value()) {
    auto table1_transitions = transition_buffer_->PopTransition();
    table1_transition_ = rela::makeBatch(table1_transitions, "cpu");

    // Update table0 transition.
    auto &table0_reward = table0_transition_.reward;
    table0_reward[-1] = dict.at("r").item<float>();
    if (table0_reward.size(0) > 1) {
      for (int i = static_cast<int>(table0_reward.size(0)) - 2; i >= 0; i--) {
        table0_reward[i] += table0_reward[i + 1] * gamma_;
      }
    }
  } else {
    auto table0_transitions = transition_buffer_->PopTransition();
    table0_transition_ = rela::makeBatch(table0_transitions, "cpu");
  }

//  std::cout << "table0 transition size: " << table0_transition_.reward.size(0) << std::endl;
  auto input0 = table0_transition_.toDict();
//  for(const auto& kv:input0){
//    std::cout << kv.first << ": " << kv.second.sizes() << std::endl;
//  }
  auto priority0 = runner_->blockCall("compute_priority", input0).at("priority");
  RELA_CHECK_EQ(priority0.size(0), table0_transition_.reward.size(0));
  for (int i = 0; i < priority0.size(0); ++i) {
    replay_buffer_->add(table0_transition_.index(i), priority0[i].item<float>());
  }
  if (duplicated.value()) {
    auto input1 = table1_transition_.toDict();
    auto priority1 = runner_->blockCall("compute_priority", input1).at("priority");
    for (int i = 0; i < priority1.size(0); ++i) {
      replay_buffer_->add(table1_transition_.index(i), priority1[i].item<float>());
    }
  }

}
void BridgeFFWDActor::SetTerminal() {
  if (transition_buffer_ != nullptr) {
    transition_buffer_->PushTerminal();
  }
}

}  // namespace rlcc