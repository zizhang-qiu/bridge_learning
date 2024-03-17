//
// Created by qzz on 2023/10/22.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PIMC_H_
#define BRIDGE_LEARNING_PLAYCC_PIMC_H_
#include <memory>
#include <utility>

#include "absl/strings/str_format.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/third_party/dds/include/dll.h"

#include "play_bot.h"
#include "resampler.h"
#include "trajectory_bidding_bot.h"

namespace ble = bridge_learning_env;

int Rollout(const ble::BridgeState& state, const ble::BridgeMove& move);

struct SearchResult {
  std::vector<ble::BridgeMove> moves;
  std::vector<int> scores;
};

std::pair<ble::BridgeMove, int> GetBestAction(const SearchResult& res);

struct PIMCConfig {
  int num_worlds;
  bool search_with_one_legal_move;
  bool verbose = false;
};

class PIMCBot final : public PlayBot {
 public:
  PIMCBot(std::shared_ptr<Resampler> resampler, const PIMCConfig cfg)
      : resampler_(std::move(resampler)), cfg_(cfg), player_id_(0) {
    SetMaxThreads(0);
  }

  PIMCBot(std::shared_ptr<Resampler> resampler, const ble::Player player_id,
          const PIMCConfig cfg)
      : resampler_(std::move(resampler)), cfg_(cfg), player_id_(player_id) {
    SetMaxThreads(0);
  }

  void SetBiddingBot(const std::shared_ptr<TrajectoryBiddingBot>& bot){
    bidding_bot_ = bot;
  }

  ble::BridgeMove Step(const ble::BridgeState& state) override;

  [[nodiscard]] SearchResult Search(const ble::BridgeState& state) const;

  std::string Name() const override {
    return absl::StrFormat("PIMC, %d worlds", cfg_.num_worlds);
  }

 private:
  std::shared_ptr<Resampler> resampler_;
  const PIMCConfig cfg_;
  const ble::Player player_id_;
  std::shared_ptr<TrajectoryBiddingBot> bidding_bot_ = nullptr;
};

void PrintSearchResult(const SearchResult& res);

#endif  // BRIDGE_LEARNING_PLAYCC_PIMC_H_
