//
// Created by qzz on 2023/12/7.
//

#ifndef BRIDGE_LEARNING_PLAYCC_ALPHA_MU_BOT_H_
#define BRIDGE_LEARNING_PLAYCC_ALPHA_MU_BOT_H_
#include "pareto_front.h"
#include "play_bot.h"
#include "resampler.h"
#include "alpha_mu_search.h"
#include "bridge_state_without_hidden_info.h"
#include "transposition_table.h"

struct AlphaMuConfig {
  int num_max_moves;
  int num_worlds;
  bool search_with_one_legal_move;
  bool use_transportation_table = true;
  bool root_cut = true;
  bool early_cut = true;

};
class VanillaAlphaMuBot final : public PlayBot {
 public:
  VanillaAlphaMuBot(const std::shared_ptr<Resampler> &resampler, const AlphaMuConfig cfg) :
      resampler_(resampler), cfg_(cfg) {}

  ble::BridgeMove Act(const ble::BridgeState &state) override;

  // Act on given worlds.
  ble::BridgeMove Act(const ble::BridgeState &state, const std::vector<ble::BridgeState> &worlds) const;

  [[nodiscard]] ParetoFront Search(const ble::BridgeState &state) const;

 private:
  std::shared_ptr<Resampler> resampler_;
  const AlphaMuConfig cfg_;
};

class AlphaMuBot final : public PlayBot {
 public:
  AlphaMuBot(const std::shared_ptr<Resampler> &resampler, const AlphaMuConfig cfg) :
      resampler_(resampler),
      cfg_(cfg),
      tt_() {}

  ble::BridgeMove Act(const ble::BridgeState &state) override;

  [[nodiscard]] ParetoFront Search(const ble::BridgeStateWithoutHiddenInfo &state, int num_max_moves,
                                   const Worlds &worlds, const ParetoFront& alpha);

 private:
  TranspositionTable tt_;
  std::shared_ptr<Resampler> resampler_;
  const AlphaMuConfig cfg_;
};
#endif // BRIDGE_LEARNING_PLAYCC_ALPHA_MU_BOT_H_
