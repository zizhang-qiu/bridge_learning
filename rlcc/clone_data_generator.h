#ifndef RLCC_CLONE_DATA_GENERATOR_H
#define RLCC_CLONE_DATA_GENERATOR_H

#include "bridge_lib/bridge_state.h"
#include "bridge_lib/utils.h"
#include "rela/prioritized_replay.h"
#include "rela/thread_loop.h"
#include "rnn_buffer.h"
#include "bridge_env.h"


namespace ble = bridge_learning_env;

namespace rlcc {
using GameTrajectory = std::vector<int>;

class DataGenLoop : public rela::ThreadLoop {
 public:
  DataGenLoop(std::shared_ptr<rela::RNNPrioritizedReplay>& replay_buffer,
              const ble::GameParameters& params,
              const std::vector<GameTrajectory>& game_trajectories, int max_len,
              bool inf_loop, int seed)
      : replay_buffer_(replay_buffer),
        params_(params),
        max_len_(max_len),
        inf_loop_(inf_loop),
        rng_(seed) {
    for (int i = 0; i < ble::kNumPlayers; ++i) {
      transition_buffers_.emplace_back(max_len_, 1.0);
    }
  }

  void mainLoop() override;

 private:
  // Transition buffers for each player.
  std::vector<RNNTransitionBuffer> transition_buffers_;
  int epoch_ = 0;
  std::mt19937 rng_;

  const int max_len_;
  const bool inf_loop_;
  const std::vector<GameTrajectory> game_trajectories_;
  const ble::GameParameters params_;
  std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer_;
};
}  // namespace rlcc

#endif /* RLCC_CLONE_DATA_GENERATOR_H */
