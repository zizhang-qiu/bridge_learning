#include "clone_data_generator.h"

#include <numeric>

namespace rlcc {
void DataGenLoop::mainLoop() {
  RELA_CHECK_GT(game_trajectories_.size(), 0);
  std::vector<size_t> idx_left;
  while (!terminated()) {
    if (idx_left.size() <= 0) {
      if (!inf_loop_) {
        if (epoch_ == 0) {
          ++epoch_;
        } else {
          break;
        }
      }
      idx_left.resize(game_trajectories_.size());
      std::iota(idx_left.begin(), idx_left.end(),0);
      std::shuffle(idx_left.begin(), idx_left.end(), rng_);
    }

    size_t idx = idx_left.back();
    idx_left.pop_back();

    const GameTrajectory game_trajectory = game_trajectories_[idx];
    BridgeEnv env{params_, {}};

    // Deal.
    for(int i=0; i<ble::kNumCards; ++i){
        const int uid = game_trajectory[i];
        env.Step(uid);
    }

    for(size_t midx=ble::kNumCards; midx < game_trajectory.size(); ++midx){
        int current_player = env.CurrentPlayer();
        
    }
  }
}
}  // namespace rlcc