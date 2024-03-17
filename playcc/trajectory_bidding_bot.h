#ifndef PLAYCC_TRAJECTORY_BIDDING_BOT_H
#define PLAYCC_TRAJECTORY_BIDDING_BOT_H
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_move.h"
#include "bridge_lib/bridge_state.h"
#include "utils.h"

// A bot make bids according to given trajectories;
class TrajectoryBiddingBot {
 public:
  TrajectoryBiddingBot(const std::vector<std::vector<int>>& trajectories,
                       const std::shared_ptr<ble::BridgeGame>& game);

  TrajectoryBiddingBot() : TrajectoryBiddingBot({}, ble::default_game) {}

  ble::BridgeMove Step(const ble::BridgeState& state);

  void AddTrajectory(const std::vector<int>& trajectory) {
    AddFromTrajectory(trajectory);
  }

 private:
  void AddFromTrajectory(const std::vector<int>& trajectory);
  std::unordered_map<std::vector<int>, int, VectorHash> map_;
  std::shared_ptr<ble::BridgeGame> game_;

  std::mutex m_;
};

#endif /* PLAYCC_TRAJECTORY_BIDDING_BOT_H */
