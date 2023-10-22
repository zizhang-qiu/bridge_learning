//
// Created by qzz on 2023/10/16.
//

#ifndef BRIDGE_LEARNING_RLCC_SUPERVISE_DATA_GENERATOR_H_
#define BRIDGE_LEARNING_RLCC_SUPERVISE_DATA_GENERATOR_H_
#include <utility>
#include <cassert>

#include "bridge_lib/canonical_encoder.h"
#include "rela/tensor_dict.h"
#include "torch/torch.h"
namespace ble = bridge_learning_env;
namespace rlcc {
class SuperviseDataGenerator {
 public:
  SuperviseDataGenerator(const std::vector<std::vector<int>> &trajectories,
                         int batch_size,
                         const std::shared_ptr<ble::BridgeGame> &game,
                         int seed) : trajectories_(trajectories),
                                     batch_size_(batch_size),
                                     game_(game),
                                     encoder_(game),
                                     rng_(seed) {}

  rela::TensorDict NextBatch(std::string &device){
    std::vector<rela::TensorDict> obs_labels;
    torch::Device d{device};
    for (int i = 0; i < batch_size_; ++i) {
      auto current_trajectory = trajectories_[index_];
      assert(current_trajectory.size()>= ble::kNumCards + ble::kNumPlayers);
      index_ = (index_ + 1) % trajectories_.size();
      auto dis = std::uniform_int_distribution<int>(ble::kNumCards, current_trajectory.size() - 1);
      auto random_index = dis(rng_);
      auto state = std::make_unique<ble::BridgeState2>(game_);
      for (int j = 0; j < ble::kNumCards; ++j) {
        auto move = game_->GetChanceOutcome(current_trajectory[j]);
        state->ApplyMove(move);
      }
      for (int j = ble::kNumCards; j < random_index; ++j) {
        auto move = game_->GetMove(current_trajectory[j]);
        state->ApplyMove(move);
      }
      const auto observation = ble::BridgeObservation(*state, state->CurrentPlayer());
      const auto encoding = encoder_.Encode(observation);
      rela::TensorDict obs_label = {
          {"s", torch::tensor(encoding, {torch::kFloat32}).to(d)},
          {"label", torch::tensor(current_trajectory[random_index]).to(d)}
      };
      obs_labels.push_back(obs_label);
    }
    return rela::tensor_dict::stack(obs_labels, 0);
  }

  rela::TensorDict AllData(std::string &device){
    std::vector<rela::TensorDict> obs_labels;
    torch::Device d{device};
    for(int idx=0; idx < trajectories_.size(); ++idx){
      auto current_trajectory = trajectories_[idx];
      assert(current_trajectory.size()>= ble::kNumCards + ble::kNumPlayers);
      auto state = std::make_unique<ble::BridgeState2>(game_);
      for (int j = 0; j < ble::kNumCards; ++j) {
        auto move = game_->GetChanceOutcome(current_trajectory[j]);
        state->ApplyMove(move);
      }
      for (int j = ble::kNumCards; j < current_trajectory.size(); ++j) {
        const auto observation = ble::BridgeObservation(*state, state->CurrentPlayer());
        const auto encoding = encoder_.Encode(observation);
        rela::TensorDict obs_label = {
            {"s", torch::tensor(encoding, {torch::kFloat32}).to(d)},
            {"label", torch::tensor(current_trajectory[j]).to(d)}
        };
        obs_labels.push_back(obs_label);
        auto move = game_->GetMove(current_trajectory[j]);
        state->ApplyMove(move);
      }

    }
    return rela::tensor_dict::stack(obs_labels, 0);
  }

 private:
  std::vector<std::vector<int>> trajectories_;
  int index_ = 0;
  const int batch_size_;
  const ble::CanonicalEncoder encoder_;
  const std::shared_ptr<ble::BridgeGame> game_;
  std::mt19937 rng_;
};
}

#endif //BRIDGE_LEARNING_RLCC_SUPERVISE_DATA_GENERATOR_H_
