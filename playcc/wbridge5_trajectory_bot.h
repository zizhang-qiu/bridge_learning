//
// Created by qzz on 2024/1/21.
//

#ifndef WBRIDGE5_TRAJECTORY_BOT_H
#define WBRIDGE5_TRAJECTORY_BOT_H

#include <unordered_map>

#include "bridge_lib/bridge_state.h"
#include "utils.h"
#include "play_bot.h"
namespace ble = bridge_learning_env;



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
