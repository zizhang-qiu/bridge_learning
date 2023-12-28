//
// Created by qzz on 2023/10/22.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PIMC_H_
#define BRIDGE_LEARNING_PLAYCC_PIMC_H_
#include <utility>

#include "absl/strings/str_format.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/third_party/dds/include/dll.h"

#include "play_bot.h"
#include "resampler.h"

namespace ble = bridge_learning_env;

int Rollout(const ble::BridgeState &state, const ble::BridgeMove &move);

struct SearchResult {
  std::vector<ble::BridgeMove> moves;
  std::vector<int> scores;
};

std::pair<ble::BridgeMove, int> GetBestAction(const SearchResult &res);

struct PIMCConfig {
  int num_worlds;
  bool search_with_one_legal_move;
};

class PIMCBot final : public PlayBot {
  public:
  PIMCBot(std::shared_ptr<Resampler> resampler, const PIMCConfig cfg) : resampler_(std::move(resampler)), cfg_(cfg) {
    SetMaxThreads(0);
  }

  ble::BridgeMove Act(const ble::BridgeState &state) override;

  ble::BridgeMove ActWithWorlds(const ble::BridgeState &state, const std::vector<ble::BridgeState> &worlds) const {
    const auto &legal_moves = state.LegalMoves();
    const int num_legal_moves = static_cast<int>(legal_moves.size());
    // Only one legal move, return it.
    if (num_legal_moves == 1) {
      return legal_moves[0];
    }
    SearchResult res{};
    res.moves = legal_moves;
    res.scores = std::vector<int>(num_legal_moves, 0);
    for (int i = 0; i < cfg_.num_worlds; ++i) {
      //      std::cout << sampled_state->ToString() << std::endl;
      for (int j = 0; j < num_legal_moves; ++j) {
        const int score = Rollout(worlds[i], legal_moves[j]);
        //        std::cout << score << std::endl;
        res.scores[j] += score;
      }
      //      std::cout << "accumulate scores" << std::endl;
    }
    auto [move, score] = GetBestAction(res);
    return move;
  }

  [[nodiscard]] SearchResult Search(const ble::BridgeState &state) const;

  std::string Name() const override { return absl::StrFormat("PIMC, %d worlds", cfg_.num_worlds); }

  private:
  std::shared_ptr<Resampler> resampler_;
  const PIMCConfig cfg_;
};

void PrintSearchResult(const SearchResult &res);

#endif // BRIDGE_LEARNING_PLAYCC_PIMC_H_
