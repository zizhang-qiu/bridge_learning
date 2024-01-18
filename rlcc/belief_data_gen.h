//
// Created by qzz on 2024/1/18.
//

#ifndef BELIEF_DATA_GEN_H
#define BELIEF_DATA_GEN_H
#include "bridge_lib/canonical_encoder.h"
#include "rela/logging.h"
#include "rela/tensor_dict.h"

namespace ble = bridge_learning_env;

std::vector<int> NoPlayTrajectory(const std::vector<int>& trajectory);

class BeliefDataGen {
  public:
    rela::TensorDict NextBatch(const std::string& device) {
      std::vector<rela::TensorDict> obs_labels;
      torch::Device d{device};
      for (int i = 0; i < batch_size_; ++i) {
        auto current_trajectory = trajectories_[index_];
        RELA_CHECK_GE(current_trajectory.back(), ble::kBiddingActionBase);
        const auto state = std::make_unique<ble::BridgeState>(game_);
        for (int j = 0; j < ble::kNumCards; ++j) {
          const auto move = game_->GetChanceOutcome(current_trajectory[j]);
          state->ApplyMove(move);
        }
        for (int j = ble::kNumCards; j < current_trajectory.size(); ++j) {
          const auto move = game_->GetMove(current_trajectory[j]);
          state->ApplyMove(move);
        }
        const auto observation = ble::BridgeObservation(*state, state->CurrentPlayer());
        const auto encoding = encoder_.Encode(observation);

        rela::TensorDict obs_label = {
          {"s", torch::tensor(encoding, {torch::kFloat32}).to(d)},

        };
        obs_labels.push_back(obs_label);
      }

    }

  private:
    std::vector<std::vector<int>> trajectories_;
    int index_ = 0;
    const int batch_size_;
    const ble::CanonicalEncoder encoder_;
    const std::shared_ptr<ble::BridgeGame> game_;
};
#endif //BELIEF_DATA_GEN_H
