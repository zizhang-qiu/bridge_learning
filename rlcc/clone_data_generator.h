#ifndef RLCC_CLONE_DATA_GENERATOR_H
#define RLCC_CLONE_DATA_GENERATOR_H

#include <utility>

#include "bridge_env.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/utils.h"
#include "rela/context.h"
#include "rela/prioritized_replay.h"
#include "rela/thread_loop.h"
#include "rnn_buffer.h"
#include "bridge_actor.h"

namespace ble = bridge_learning_env;

namespace rlcc {
using GameTrajectory = std::vector<int>;

class DataGenLoop : public rela::ThreadLoop {
 public:
  DataGenLoop(std::shared_ptr<rela::RNNPrioritizedReplay> &replay_buffer,
              ble::GameParameters params,
              BridgeEnvOptions env_options,
              const std::vector<GameTrajectory> &game_trajectories,
              int max_len,
              bool inf_loop,
              int seed,
              std::string_view reward_type,
              int thread_idx)
      : replay_buffer_(replay_buffer),
        game_trajectories_(game_trajectories),
        params_(std::move(params)),
        env_options_(std::move(env_options)),
        max_len_(max_len),
        inf_loop_(inf_loop),
        rng_(seed),
        reward_type_(reward_type),
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
  std::string reward_type_;
  const std::vector<GameTrajectory> game_trajectories_;
  const ble::GameParameters params_;
  BridgeEnvOptions env_options_;
  std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer_;

  int thread_idx_;
};

class CloneDataGenerator {
 public:
  CloneDataGenerator(std::shared_ptr<rela::RNNPrioritizedReplay> &replay_buffer,
                     int max_len, int num_threads,
                     std::string_view reward_type)
      : replay_buffer_(replay_buffer),
        max_len_(max_len),
        num_threads_(num_threads),
        game_trajectories_(num_threads),
        reward_type_(reward_type) {}

  void SetGameParams(const ble::GameParameters &params) {
    game_params_ = params;
  }

  void SetEnvOptions(const BridgeEnvOptions &env_options) {
    env_options_ = env_options;
  }

  void SetRewardType(const std::string &reward_type) {
    reward_type_ = reward_type;
  }

  void AddGame(const GameTrajectory &game_trajectory) {
    game_trajectories_[next_game_thread_].push_back(game_trajectory);
    next_game_thread_ = (next_game_thread_ + 1) % num_threads_;
  }

  void StartDataGeneration(bool inf_loop, int seed);

  void Terminate() { context_ = nullptr; }

  [[nodiscard]] std::vector<rela::RNNTransition> GenerateEvalData(
      int batch_size, const std::string &device,
      const std::vector<GameTrajectory> &game_trajectories);

 private:
  ble::GameParameters game_params_;
  BridgeEnvOptions env_options_;
  const int num_threads_;
  const int max_len_;
  std::string reward_type_;

  std::unique_ptr<rela::Context> context_;
  std::vector<std::shared_ptr<DataGenLoop>> threads_;
  std::vector<std::vector<GameTrajectory>> game_trajectories_;
  std::shared_ptr<rela::RNNPrioritizedReplay> replay_buffer_;

  int next_game_thread_ = 0;
};

class FFDataGenLoop : public rela::ThreadLoop {
 public:
  FFDataGenLoop(std::shared_ptr<rela::FFPrioritizedReplay> &replay_buffer,
                ble::GameParameters params,
                BridgeEnvOptions env_options,
                const std::vector<GameTrajectory> &game_trajectories,
                bool inf_loop,
                int seed,
                std::string_view reward_type,
                float gamma,
                int thread_idx)
      : replay_buffer_(replay_buffer),
        game_trajectories_(game_trajectories),
        params_(std::move(params)),
        env_options_(std::move(env_options)),
        inf_loop_(inf_loop),
        rng_(seed),
        reward_type_(reward_type),
        thread_idx_(thread_idx) {
    for (int i = 0; i < ble::kNumPlayers; ++i) {
      transition_buffers_.emplace_back(gamma);
    }
  }

  void mainLoop() override;
 private:
  // Transition buffers for each player.
  std::vector<FFTransitionBuffer> transition_buffers_;
  int epoch_ = 0;
  std::mt19937 rng_;

  const bool inf_loop_;
  std::string reward_type_;
  const std::vector<GameTrajectory> game_trajectories_;
  const ble::GameParameters params_;
  BridgeEnvOptions env_options_;
  std::shared_ptr<rela::FFPrioritizedReplay> replay_buffer_;

  int thread_idx_;
};

class FFCloneDataGenerator {
 public:
  FFCloneDataGenerator(std::shared_ptr<rela::FFPrioritizedReplay> &replay_buffer,
                       int num_threads,
                       const BridgeEnvOptions &env_options,
                       std::string_view reward_type,
                       float gamma)
      : replay_buffer_(replay_buffer),
        num_threads_(num_threads),
        env_options_(env_options),
        reward_type_(reward_type),
        gamma_(gamma),
        game_trajectories_(num_threads){}

  void SetGameParams(const ble::GameParameters &params) {
    game_params_ = params;
  }

  void SetEnvOptions(const BridgeEnvOptions &env_options) {
    env_options_ = env_options;
  }

  void SetRewardType(const std::string &reward_type) {
    reward_type_ = reward_type;
  }

  void AddGame(const GameTrajectory &game_trajectory) {
    game_trajectories_[next_game_thread_].push_back(game_trajectory);
    next_game_thread_ = (next_game_thread_ + 1) % num_threads_;
  }

  void StartDataGeneration(bool inf_loop, int seed);

  void Terminate() { context_ = nullptr; }

  [[nodiscard]] std::vector<rela::FFTransition> GenerateEvalData(
      int batch_size, const std::string &device,
      const std::vector<GameTrajectory> &game_trajectories);

 private:
  ble::GameParameters game_params_;
  BridgeEnvOptions env_options_;
  const int num_threads_;
  std::string reward_type_;
  float gamma_;

  std::unique_ptr<rela::Context> context_;
  std::vector<std::shared_ptr<FFDataGenLoop>> threads_;
  std::vector<std::vector<GameTrajectory>> game_trajectories_;
  std::shared_ptr<rela::FFPrioritizedReplay> replay_buffer_;

  int next_game_thread_ = 0;
};
}  // namespace rlcc

#endif /* RLCC_CLONE_DATA_GENERATOR_H */
