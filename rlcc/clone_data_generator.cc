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
      std::iota(idx_left.begin(), idx_left.end(), 0);
      std::shuffle(idx_left.begin(), idx_left.end(), rng_);
    }

    size_t idx = idx_left.back();
    idx_left.pop_back();
    // std::cout << "Thread " << thread_idx_ << " Get idx\n";

    const GameTrajectory game_trajectory = game_trajectories_[idx];
    BridgeEnv env{params_, {}};

    // std::cout << "Thread " << thread_idx_ << " current trajectory:\n";
    // for(const int action:game_trajectory){
    //   std::cout << action << ", ";
    // }
    // std::cout << "\n";

    // Deal.
    env.ResetWithDeckAndDoubleDummyResults(
        std::vector<int>(game_trajectory.begin(),
                         game_trajectory.begin() + ble::kNumCards),
        std::vector<int>(
            kDoubleDummyResultSize,
            0));  // Fake double dummy table here since we don't want to compute score.

    // std::cout << "Thread " << thread_idx_ << " Reset env\n";

    const size_t end_size =
        game_trajectory.size() > ble::kNumCards + ble::kNumPlayers
            ? game_trajectory.size() - ble::kNumCards
            : game_trajectory.size();
    for (size_t midx = ble::kNumCards; midx < end_size; ++midx) {
      int current_player = env.CurrentPlayer();
      // std::cout << "Thread " << thread_idx_
      //           << " Get cur player: " << current_player << std::endl;
      for (int player = 0; player < ble::kNumPlayers; ++player) {
        auto feature = env.Feature(player);
        // Split public and private.
        const int kPrivateFeatureSize = feature.at("s").size(0) - ble::kNumCards;
        feature["publ_s"] = feature.at("s").index(
            {torch::indexing::Slice(0, kPrivateFeatureSize)});
        feature["priv_s"] = feature.at("s");
        transition_buffers_[player].PushObs(feature);
        // std::cout << "Thread " << thread_idx_
        //         << " Push obs.\n";
        int action;
        if (player == current_player) {
          action = game_trajectory[midx];
        } else {
          action = env.NoOPUid();
        }
        transition_buffers_[player].PushAction({{"a", torch::tensor(action)}});
        // std::cout << "Thread " << thread_idx_
        //         << " Push action.\n";
      }

      env.Step(game_trajectory[midx]);

      if (midx == end_size - 1) {
        auto rewards = env.Rewards();
        for (int player = 0; player < ble::kNumPlayers; ++player) {
          transition_buffers_[player].PushTerminal();
          transition_buffers_[player].PushReward(
              {{"r", torch::tensor(rewards[player])}});
        }
        // std::cout << "Thread " << thread_idx_
        //         << " Push reward and terminal.\n";
      }
    }
    // std::cout << "Thread " << thread_idx_ << "Simulate game done\n";

    for (int player = 0; player < ble::kNumPlayers; ++player) {
      replay_buffer_->add(transition_buffers_[player].PopTransition(), 1.0);
    }
  }
}

void CloneDataGenerator::StartDataGeneration(bool inf_loop, int seed) {
  // for (int i = 0; i < num_threads_; ++i) {
  //   std::cout << i << ": " << game_trajectories_[i].size() << "\n";
  // }
  std::mt19937 rng(seed);
  context_ = std::make_unique<rela::Context>();
  for (int i = 0; i < num_threads_; ++i) {
    int seed = static_cast<int>(rng());
    auto thread = std::make_shared<DataGenLoop>(replay_buffer_, game_params_,
                                                game_trajectories_[i], max_len_,
                                                inf_loop, seed, i);
    context_->pushThreadLoop(thread);
    threads_.push_back(thread);
  }
  context_->start();
}

std::vector<rela::RNNTransition> CloneDataGenerator::GenerateEvalData(
    int batch_size, const std::string& device,
    const std::vector<GameTrajectory>& game_trajectories) const {
  const int num_games = static_cast<int>(game_trajectories.size());

  std::vector<rela::RNNTransition> batches;
  // Transition buffers for each player.
  std::vector<RNNTransitionBuffer> transition_buffers_;
  for (int i = 0; i < ble::kNumPlayers; ++i) {
    transition_buffers_.emplace_back(max_len_, 1.0);
  }
  std::vector<rela::RNNTransition> v_transitions;

  for (int i = 0; i < num_games; ++i) {
    // std::cout << "i=" << i << std::endl;
    const std::vector<int> game_trajectory = game_trajectories[i];
    const size_t end_size =
        game_trajectory.size() > ble::kNumCards + ble::kNumPlayers
            ? game_trajectory.size() - ble::kNumCards
            : game_trajectory.size();

    BridgeEnv env{game_params_, {}};
    // Deal.
    env.ResetWithDeckAndDoubleDummyResults(
        std::vector<int>(game_trajectory.begin(),
                         game_trajectory.begin() + ble::kNumCards),
        std::vector<int>(
            kDoubleDummyResultSize,
            0));  // Fake double dummy table here since we don't want to compute score.

    // std::cout << "Reset env" << std::endl;

    for (size_t midx = ble::kNumCards; midx < end_size; ++midx) {
      const int current_player = env.CurrentPlayer();
      for (int player = 0; player < ble::kNumPlayers; ++player) {
        auto feature = env.Feature(player);
        // Split public and private.
        const int kPrivateFeatureSize = feature.at("s").size(0) - ble::kNumCards;
        feature["publ_s"] = feature.at("s").index(
            {torch::indexing::Slice(0, kPrivateFeatureSize)});
        feature["priv_s"] = feature.at("s");
        transition_buffers_[player].PushObs(feature);

        int action;
        if (player == current_player) {
          action = game_trajectory[midx];
        } else {
          action = env.NoOPUid();
        }
        transition_buffers_[player].PushAction({{"a", torch::tensor(action)}});
      }

      env.Step(game_trajectory[midx]);
      if (midx == end_size - 1) {
        auto rewards = env.Rewards();
        for (int player = 0; player < ble::kNumPlayers; ++player) {
          transition_buffers_[player].PushTerminal();
          transition_buffers_[player].PushReward(
              {{"r", torch::tensor(rewards[player])}});
        }
      }
    }

    for (int player = 0; player < ble::kNumPlayers; ++player) {
      v_transitions.push_back(transition_buffers_[player].PopTransition());
      if (v_transitions.size() % batch_size == 0) {
        rela::RNNTransition transition = rela::makeBatch(v_transitions, device);
        batches.push_back(transition);
        v_transitions.clear();
      }
    }
  }

  return batches;
}
}  // namespace rlcc