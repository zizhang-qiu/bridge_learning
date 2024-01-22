//
// Created by qzz on 2024/1/21.
//

#ifndef WBRIDGE5_TRAJECTORY_BOT_H
#define WBRIDGE5_TRAJECTORY_BOT_H

#include <unordered_map>

#include "bridge_lib/bridge_state.h"

#include "play_bot.h"
namespace ble = bridge_learning_env;

// Custom hash function for std::vector<int>
struct VectorHash {
  std::size_t operator()(const std::vector<int>& vec) const {
    std::hash<int> hasher;
    std::size_t hash_value = 0;

    for (const int& element : vec) {
      hash_value ^= hasher(element) + 0x9e3779b9 + (hash_value << 6) + (
        hash_value >> 2);
    }

    return hash_value;
  }
};

class WBridge5TrajectoryBot : public PlayBot {
  public:
    WBridge5TrajectoryBot(const std::vector<std::vector<int>>& trajectories,
                          const std::shared_ptr<ble::BridgeGame>& game) {
      for (const auto& trajectory : trajectories) {
        SPIEL_CHECK_GT(trajectory.size(), game->MinGameLength());

        std::vector<int> deal_and_bidding_trajectory(
            trajectory.begin(), trajectory.end() - ble::kNumCards);
        // for (const int uid : deal_and_bidding_trajectory) {
        //   std::cout << uid << ", " << std::endl;
        // }
        // std::cout << std::endl;
        const int move_uid = trajectory[trajectory.size() - ble::kNumCards];
        // std::cout << "move uid: " << move_uid << std::endl;
        traj_uid_map_[deal_and_bidding_trajectory] = move_uid;
      }
      SPIEL_CHECK_EQ(traj_uid_map_.size(), trajectories.size());
    }

    ble::BridgeMove Step(const ble::BridgeState& state) override {
      if (!traj_uid_map_.count(state.UidHistory())) {
        return state.LegalMoves()[0];
      }
      const int uid = traj_uid_map_[state.UidHistory()];
      const auto move = state.ParentGame()->GetMove(uid);
      return move;
    }

  private:
    std::unordered_map<std::vector<int>, int, VectorHash> traj_uid_map_;
};
#endif //WBRIDGE5_TRAJECTORY_BOT_H
