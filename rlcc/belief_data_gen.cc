//
// Created by qzz on 2024/1/18.
//

#include "belief_data_gen.h"

#include "rela/utils.h"

std::vector<int> NoPlayTrajectory(const std::vector<int>& trajectory) {
  if (const size_t size = trajectory.size();
    size == ble::kNumCards + ble::kNumPlayers) {
    return trajectory;
  }
  return std::vector<int>(trajectory.begin(),
                          trajectory.end() - ble::kNumCards);
}

rela::TensorDict BeliefDataGen::NextBatch(const std::string& device) {
  std::vector<rela::TensorDict> obs_labels;
  torch::Device d{device};
  for (int i = 0; i < batch_size_; ++i) {
    if (cached_data_[index_].empty()) {
      // If we didn't get the feature and belief, get them and cache.
      auto current_trajectory = trajectories_[index_];
      rela::TensorDict obs_label = GetDataFromTrajectory(
          current_trajectory, device);
      obs_labels.push_back(obs_label);
      cached_data_[index_] = obs_label;
    } else {
      // If we already have features, use it.
      obs_labels.push_back(
          rela::tensor_dict::toDevice(cached_data_[index_], device));
    }
    // Update index.
    index_ = (index_ + 1) % static_cast<int>(trajectories_.size());
  }
  return rela::tensor_dict::stack(obs_labels, 0);
}

rela::TensorDict BeliefDataGen::AllData(const std::string& device) {
  std::vector<rela::TensorDict> obs_labels;
  torch::Device d{device};
  for (int i = 0; i < static_cast<int>(trajectories_.size()); ++i) {
    if (cached_data_[i].empty()) {
      // If we didn't get the feature and belief, get them and cache.

      // rela::utils::printVector(trajectories_[i]);
      rela::TensorDict obs_label = GetDataFromTrajectory(
          trajectories_[i], device);

      cached_data_[i] = obs_label;
      obs_labels.push_back(obs_label);
    } else {
      // If we already have features, use it.
      obs_labels.
          push_back(rela::tensor_dict::toDevice(cached_data_[i], device));
    }
  }
  return rela::tensor_dict::stack(obs_labels, 0);
}

rela::TensorDict BeliefDataGen::GetDataFromTrajectory(
    const std::vector<int>& trajectory,
    const std::string& device) const {
  // RELA_CHECK_GE(trajectory.back(), ble::kBiddingActionBase);

  torch::Device d{device};

  auto state = std::make_unique<ble::BridgeState>(game_);

  for (int j = 0; j < ble::kNumCards; ++j) {
    const auto move = game_->GetChanceOutcome(trajectory[j]);
    state->ApplyMove(move);
  }

  for (int j = ble::kNumCards; j < trajectory.size(); ++j) {
    const auto move = game_->GetMove(trajectory[j]);
    state->ApplyMove(move);
  }

  const auto observation = ble::BridgeObservation(
      *state, state->CurrentPlayer());
  auto encoding = encoder_.Encode(observation);
  encoding = std::vector<int>(encoding.begin(),
                              encoding.begin()
                              + encoder_.Shape()[0]);
  const auto label = encoder_.EncodeOtherHands(observation);
  const auto belief_he_one_hot = encoder_.EncodeOtherHandEvaluationsOneHot(observation);
  const auto belief_he = encoder_.EncodeOtherHandEvaluations(observation);
  rela::TensorDict obs_label = {
      {"s", torch::tensor(encoding, {torch::kFloat32}).to(d)},
      {"belief", torch::tensor(label, {torch::kFloat32}).to(d)},
      {"belief_he_one_hot", torch::tensor(belief_he, {torch::kFloat32}).to(d)},
    {"belief_he", torch::tensor(belief_he, {torch::kFloat32}).to(d)}
  };
  return obs_label;
}
