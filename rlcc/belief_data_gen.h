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
    BeliefDataGen(const std::vector<std::vector<int>>& trajectories,
                  const int batch_size,
                  const std::shared_ptr<ble::BridgeGame>& game)
      : trajectories_(trajectories),
        cached_data_(trajectories.size()),
        batch_size_(batch_size),
        game_(game),
        encoder_(game_) {
    }

    // Get next batch.
    rela::TensorDict NextBatch(const std::string& device);

    // Get all data from trajectories, should not use when data is large.
    [[nodiscard]] rela::TensorDict AllData(const std::string& device);

  private:
    std::vector<std::vector<int>> trajectories_;
    std::vector<rela::TensorDict> cached_data_;
    // A trajectory is available if it's not passed out.
    int index_ = 0;
    const int batch_size_;
    const std::shared_ptr<ble::BridgeGame> game_;
    const ble::CanonicalEncoder encoder_;

    [[nodiscard]] rela::TensorDict GetDataFromTrajectory(const std::vector<int>& trajectory,
                                                         const std::string& device) const;
};
#endif //BELIEF_DATA_GEN_H
