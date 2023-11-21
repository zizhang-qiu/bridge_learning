//
// Created by qzz on 2023/10/22.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PIMC_H_
#define BRIDGE_LEARNING_PLAYCC_PIMC_H_
#include <utility>

#include "bridge_lib/bridge_state.h"
#include "resampler.h"
#include "play_bot.h"
//#include "rela/logging.h"
#include "bridge_lib/third_party/dds/include/dll.h"
namespace ble = bridge_learning_env;


int Rollout(const ble::BridgeState &state, const ble::BridgeMove &move);

struct SearchResult {
  std::vector<ble::BridgeMove> moves;
  std::vector<int> scores;
};

std::pair<ble::BridgeMove, int> GetBestAction(const SearchResult &res);

class PIMCBot final : public PlayBot{
 public:
  PIMCBot(std::shared_ptr<Resampler> resampler, const int num_sample)
      : resampler_(std::move(resampler)), num_sample_(num_sample) {
    SetMaxThreads(0);
  }

  ble::BridgeMove Act(const ble::BridgeState &state) override{
    const SearchResult res = Search(state);
    auto [move, score] = GetBestAction(res);
    return move;
  }

  [[nodiscard]] SearchResult Search(const ble::BridgeState &state) const;

  private:
  std::shared_ptr<Resampler> resampler_;
  int num_sample_;

};

void PrintSearchResult(const SearchResult &res);

#endif //BRIDGE_LEARNING_PLAYCC_PIMC_H_
