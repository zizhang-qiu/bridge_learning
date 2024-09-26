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
    if (reward_type_ == "dds") {
      env_options_.playing_phase = false;
    } else {
      env_options_.playing_phase = true;
    }
    BridgeEnv env{params_, env_options_};

    // Deal.

    env.ResetWithDeck(std::vector<int>(game_trajectory.begin(),
                                       game_trajectory.begin() + ble::kNumCards));



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
        feature.erase("s");
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
        for (int player = 0; player < ble::kNumPlayers; ++player) {
          transition_buffers_[player].PushTerminal();
        }
      }
    }
    std::vector<float> rewards;
    if (reward_type_ == "dds") {
      rewards = env.Rewards();
    } else {
      // Rollout the playing phase to get rewards.
      if (env.Terminated()) {
        // May be passed out.
        rewards = env.Rewards();
      } else {
        for (size_t i = end_size; i < game_trajectory.size(); ++i) {
          const int uid = game_trajectory.at(i);
          env.Step(uid);
        }
        RELA_CHECK(env.Terminated());
        rewards = env.Rewards();
      }
    }
    for (int player = 0; player < ble::kNumPlayers; ++player) {
      transition_buffers_[player].PushReward(
          {{"r", torch::tensor(rewards[player])}});
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
    auto thread = std::make_shared<DataGenLoop>(replay_buffer_,
                                                game_params_,
                                                env_options_,
                                                game_trajectories_[i],
                                                max_len_,
                                                inf_loop, seed,
                                                reward_type_, i);
    context_->pushThreadLoop(thread);
    threads_.push_back(thread);
  }
  context_->start();
}

std::vector<rela::RNNTransition> CloneDataGenerator::GenerateEvalData(
    int batch_size, const std::string &device,
    const std::vector<GameTrajectory> &game_trajectories) {
  const int num_games = static_cast<int>(game_trajectories.size());

  std::vector<rela::RNNTransition> batches;
  // Transition buffers for each player.
  std::vector<RNNTransitionBuffer> transition_buffers_;
  transition_buffers_.reserve(ble::kNumPlayers);
  for (int i = 0; i < ble::kNumPlayers; ++i) {
    transition_buffers_.emplace_back(max_len_, 1.0);
  }
  std::vector<rela::RNNTransition> v_transitions;

  for (int i = 0; i < num_games; ++i) {
    // std::cout << "i=" << i << std::endl;
    const std::vector<int> &game_trajectory = game_trajectories[i];
    const size_t end_size =
        game_trajectory.size() > ble::kNumCards + ble::kNumPlayers
        ? game_trajectory.size() - ble::kNumCards
        : game_trajectory.size();

    if (reward_type_ == "dds") {
      env_options_.playing_phase = false;
    } else {
      env_options_.playing_phase = true;
    }

    BridgeEnv env{game_params_, env_options_};
    // Deal.
    env.ResetWithDeck(std::vector<int>(game_trajectory.begin(),
                                       game_trajectory.begin() + ble::kNumCards));

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
        feature.erase("s");
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
//        auto rewards = env.Rewards();
        for (int player = 0; player < ble::kNumPlayers; ++player) {
          transition_buffers_[player].PushTerminal();
//          transition_buffers_[player].PushReward(
//              {{"r", torch::tensor(rewards[player])}});
        }
      }
    }

    std::vector<float> rewards;
    if (reward_type_ == "dds") {
      rewards = env.Rewards();
    } else {
      // Rollout the playing phase to get rewards.
      if (env.Terminated()) {
        // May be passed out.
        rewards = env.Rewards();
      } else {
        for (size_t j = end_size; j < game_trajectory.size(); ++j) {
          const int uid = game_trajectory.at(j);
          env.Step(uid);
        }
        RELA_CHECK(env.Terminated());
        rewards = env.Rewards();
      }
    }
    for (int player = 0; player < ble::kNumPlayers; ++player) {
      transition_buffers_[player].PushReward(
          {{"r", torch::tensor(rewards[player])}});
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

void FFDataGenLoop::mainLoop() {
  RELA_CHECK_GT(game_trajectories_.size(), 0);
  std::vector<size_t> idx_left;
  if (reward_type_ == "dds") {
    env_options_.playing_phase = false;
  } else {
    env_options_.playing_phase = true;
  }
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
//    std::cout << "get idx" << std::endl;

    const GameTrajectory game_trajectory = game_trajectories_[idx];

    BridgeEnv env{params_, env_options_};

    // Deal.

    env.ResetWithDeck(std::vector<int>(game_trajectory.begin(),
                                       game_trajectory.begin() + ble::kNumCards));

    const size_t end_size =
        game_trajectory.size() > ble::kNumCards + ble::kNumPlayers
        ? game_trajectory.size() - ble::kNumCards
        : game_trajectory.size();

    for (size_t midx = ble::kNumCards; midx < end_size; ++midx) {
//      std::cout << "midx: " << midx << std::endl;
      int current_player = env.CurrentPlayer();
      auto feature = env.Feature(current_player);
      const auto [_, priv_size, publ_size] = env.FeatureSize();
      feature["perf_s"] = feature.at("s");
      feature["publ_s"] = feature.at("s").index(
          {torch::indexing::Slice(0, publ_size)});
      feature["priv_s"] = feature.at("s").index(
          {torch::indexing::Slice(0, priv_size)});
      feature.erase("s");
      feature["legal_move"] = feature.at("legal_move").index(
          {torch::indexing::Slice(ble::kNumCards, -1)}
      );
      transition_buffers_[current_player].PushObs(feature);
      const int action = game_trajectory[midx];
      transition_buffers_[current_player].PushAction({{"a", torch::tensor(action, {torch::kInt32})}});
      env.Step(action);
    }

    std::vector<float> rewards;
    if (reward_type_ == "dds") {
      rewards = env.Rewards();
    } else {
      // Rollout the playing phase to get rewards.
      if (env.Terminated()) {
        // May be passed out.
        rewards = env.Rewards();
      } else {
        for (size_t i = end_size; i < game_trajectory.size(); ++i) {
          const int uid = game_trajectory.at(i);
          env.Step(uid);
        }
        RELA_CHECK(env.Terminated());
        rewards = env.Rewards();
      }
    }
//    std::cout << "get reward." << std::endl;

    for (int player = 0; player < ble::kNumPlayers; ++player) {
      transition_buffers_[player].PushReward(rewards[player]);
      transition_buffers_[player].PushTerminal();
    }
//    std::cout << "push r and t\n";

    for (int player = 0; player < ble::kNumPlayers; ++player) {
      const auto transitions = transition_buffers_[player].PopTransition();
      for (const auto &transition : transitions) {
        replay_buffer_->add(transition, 1.0);
      }
    }
//    std::cout << "add to buffer\n";
  }
}

void FFCloneDataGenerator::StartDataGeneration(bool inf_loop, int seed) {
  std::mt19937 rng(seed);
  context_ = std::make_unique<rela::Context>();
  for (int i = 0; i < num_threads_; ++i) {
    int seed = static_cast<int>(rng());
    auto thread = std::make_shared<FFDataGenLoop>(replay_buffer_,
                                                  game_params_,
                                                  env_options_,
                                                  game_trajectories_[i],
                                                  inf_loop, seed,
                                                  reward_type_,
                                                  gamma_,
                                                  i);
    context_->pushThreadLoop(thread);
    threads_.push_back(thread);
  }
  context_->start();
}
std::vector<rela::FFTransition> FFCloneDataGenerator::GenerateEvalData(int batch_size,
                                                                       const std::string &device,
                                                                       const std::vector<GameTrajectory> &game_trajectories) {
  const int num_games = static_cast<int>(game_trajectories.size());
  std::vector<rela::FFTransition> all_transitions;
  // Transition buffers for each player.
  std::vector<FFTransitionBuffer> transition_buffers_;
  transition_buffers_.reserve(ble::kNumPlayers);
  for (int i = 0; i < ble::kNumPlayers; ++i) {
    transition_buffers_.emplace_back(gamma_);
  }
  if (reward_type_ == "dds") {
    env_options_.playing_phase = false;
  } else {
    env_options_.playing_phase = true;
  }

  for (size_t idx = 0; idx < game_trajectories.size(); ++idx) {
//    std::cout << "game idx: " << idx << std::endl;
    const GameTrajectory &game_trajectory = game_trajectories[idx];
//    std::cout << "trajectory: \n";
//    rela::utils::printVector(game_trajectory);

    BridgeEnv env{game_params_, env_options_};

    // Deal.

    env.ResetWithDeck(std::vector<int>(game_trajectory.begin(),
                                       game_trajectory.begin() + ble::kNumCards));

    const size_t end_size =
        game_trajectory.size() > ble::kNumCards + ble::kNumPlayers
        ? game_trajectory.size() - ble::kNumCards
        : game_trajectory.size();

    for (size_t midx = ble::kNumCards; midx < end_size; ++midx) {
//      std::cout << "mid idx: " << midx << std::endl;
      int current_player = env.CurrentPlayer();
      auto feature = env.Feature(current_player);
      const auto [_, priv_size, publ_size] = env.FeatureSize();
      feature["perf_s"] = feature.at("s");
      feature["publ_s"] = feature.at("s").index(
          {torch::indexing::Slice(0, publ_size)});
      feature["priv_s"] = feature.at("s").index(
          {torch::indexing::Slice(0, priv_size)});
      feature.erase("s");
      feature["legal_move"] = feature.at("legal_move").index(
          {torch::indexing::Slice(ble::kNumCards, -1)}
      );
      transition_buffers_[current_player].PushObs(feature);
      const int action = game_trajectory[midx];
      transition_buffers_[current_player].PushAction({{"a", torch::tensor(action, {torch::kInt32})}});
      env.Step(action);
    }
//    std::cout << "env:\n" << env.ToString() << std::endl;
//    std::cout << "terminated? " << env.Terminated() << std::endl;

    std::vector<float> rewards;
    if (reward_type_ == "dds") {
      rewards = env.Rewards();
    } else {
      // Rollout the playing phase to get rewards.
      if (env.Terminated()) {
        // May be passed out.
        rewards = env.Rewards();
      } else {
        for (size_t i = end_size; i < game_trajectory.size(); ++i) {
          const int uid = game_trajectory.at(i);
          env.Step(uid);
        }
        RELA_CHECK(env.Terminated());
        rewards = env.Rewards();
      }
    }
//    std::cout << "get reward.\n";
//    std::cout << "reward: ";
//    rela::utils::printVector(rewards);

    for (int player = 0; player < ble::kNumPlayers; ++player) {
      transition_buffers_[player].PushReward(rewards[player]);
      transition_buffers_[player].PushTerminal();
    }
//    std::cout << "push reward and terminal.\n";

    for (int player = 0; player < ble::kNumPlayers; ++player) {
      const auto transitions =
          transition_buffers_[player].PopTransition();
      for (const auto &transition : transitions) {
        all_transitions.push_back(transition);
      }
    }
  }

  RELA_CHECK_GT(all_transitions.size(), 0);
//  std::cout << "num all transitions: " << all_transitions.size() << std::endl;
  std::vector<rela::FFTransition> batches;
  int num_transitions = static_cast<int>(all_transitions.size());
  int num_batches = std::ceil(static_cast<float>(num_transitions) / static_cast<float>(batch_size));
  for (int i = 0; i < num_batches; ++i) {
    int left = i * batch_size;
    int right = std::min(left + batch_size, num_transitions);
    std::vector<rela::FFTransition> transition_batch(
        std::vector<rela::FFTransition>(all_transitions.begin() + left,
                                        all_transitions.begin() + right)
    );
    rela::FFTransition batch = rela::makeBatch(transition_batch, device);
    batches.push_back(batch);
  }

  return batches;

}
}  // namespace rlcc