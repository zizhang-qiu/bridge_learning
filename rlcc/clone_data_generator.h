#ifndef RLCC_CLONE_DATA_GENERATOR_H
#define RLCC_CLONE_DATA_GENERATOR_H

#include "bridge_env.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/utils.h"
#include "rela/context.h"
#include "rela/prioritized_replay.h"
#include "rela/thread_loop.h"
#include "rnn_buffer.h"

namespace ble = bridge_learning_env;

namespace rlcc {
using GameTrajectory = std::vector<int>;

class DataGenLoop : public rela::ThreadLoop {
 public:
  DataGenLoop(std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer,
              const ble::GameParameters& params,
              const std::vector<GameTrajectory>& game_trajectories, int max_len,
              bool inf_loop, int seed, int thread_idx)
      : replay_buffer_(replay_buffer),
        game_trajectories_(game_trajectories),
        params_(params),
        max_len_(max_len),
        inf_loop_(inf_loop),
        rng_(seed),
        thread_idx_(thread_idx) {
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

  int thread_idx_;
};

class CloneDataGenerator {
 public:
  CloneDataGenerator(std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer,
                     int max_len, int num_threads)
      : replay_buffer_(replay_buffer),
        max_len_(max_len),
        num_threads_(num_threads),
        game_trajectories_(num_threads) {}

  void SetGameParams(const ble::GameParameters& params) {
    game_params_ = params;
  }

  void AddGame(const GameTrajectory& game_trajectory) {
    game_trajectories_[next_game_thread_].push_back(game_trajectory);
    next_game_thread_ = (next_game_thread_ + 1) % num_threads_;
  }

  void StartDataGeneration(bool inf_loop, int seed);

  void Terminate() { context_ = nullptr; }

  std::vector<rela::RNNTransition> GenerateEvalData(
      int batch_size, const std::string& device,
      const std::vector<GameTrajectory>& game_trajectories) const;

 private:
  ble::GameParameters game_params_;
  const int num_threads_;
  const int max_len_;

  std::unique_ptr<rela::Context> context_;
  std::vector<std::shared_ptr<DataGenLoop>> threads_;
  std::vector<std::vector<GameTrajectory>> game_trajectories_;
  std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer_;

  int next_game_thread_ = 0;
};
}  // namespace rlcc

#endif /* RLCC_CLONE_DATA_GENERATOR_H */
