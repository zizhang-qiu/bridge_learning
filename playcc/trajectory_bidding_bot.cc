#include "trajectory_bidding_bot.h"
#include <mutex>
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_utils.h"
#include "playcc/common_utils/log_utils.h"

TrajectoryBiddingBot::TrajectoryBiddingBot(
    const std::vector<std::vector<int>>& trajectories,
    const std::shared_ptr<ble::BridgeGame>& game)
    : game_(game) {
  for (const auto& trajectory : trajectories) {
    AddFromTrajectory(trajectory);
  }
}

ble::BridgeMove TrajectoryBiddingBot::Step(const ble::BridgeState& state) {
  const auto trajectory = state.UidHistory();
  int uid;
  if (map_.count(trajectory)) {
    uid = map_[trajectory];
  } else {
    std::cout << "Warning, the bot doesn't know what to act at current state, "
                 "please check your trajectories."
              << std::endl;
    uid = ble::kBiddingActionBase;
  }
  const auto move = game_->GetMove(uid);
  return move;
}

void TrajectoryBiddingBot::AddFromTrajectory(
    const std::vector<int>& trajectory) {
  std::lock_guard<std::mutex> lk(m_);
  // Check if the game is complete.
  SPIEL_CHECK_GE(trajectory.size(), game_->MinGameLength());
  ble::BridgeState state{game_};
  for (int i = 0; i < ble::kNumCards; ++i) {
    const int uid = trajectory[i];
    const auto move = game_->GetChanceOutcome(uid);
    state.ApplyMove(move);
  }

  // For each auction, build a key-value pair.
  int idx = ble::kNumCards;
  while (state.IsInPhase(ble::Phase::kAuction)) {
    const int uid = trajectory[idx];
    std::vector<int> current_trajectory(trajectory.begin(),
                                        trajectory.begin() + idx);
    map_[current_trajectory] = uid;
    const auto move = game_->GetMove(uid);
    state.ApplyMove(move);
    ++idx;
  }
  
}
