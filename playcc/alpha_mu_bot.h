//
// Created by qzz on 2023/12/7.
//

#ifndef BRIDGE_LEARNING_PLAYCC_ALPHA_MU_BOT_H_
#define BRIDGE_LEARNING_PLAYCC_ALPHA_MU_BOT_H_
#include "absl/strings/str_format.h"

#include "alpha_mu_search.h"
#include "bridge_state_without_hidden_info.h"
#include "pareto_front.h"
#include "play_bot.h"
#include "resampler.h"
#include "transposition_table.h"

struct AlphaMuConfig {
  int num_max_moves;
  int num_worlds;
  bool search_with_one_legal_move = false;
  bool use_transportation_table = true;
  bool root_cut = true;
  bool early_cut = true;
};

class VanillaAlphaMuBot final : public PlayBot {
  public:
    VanillaAlphaMuBot(const std::shared_ptr<Resampler>& resampler, const AlphaMuConfig cfg) : resampler_(resampler),
      cfg_(cfg) {
    }

    ble::BridgeMove Step(const ble::BridgeState& state) override;

    // Act on given worlds.
    ble::BridgeMove ActWithWorlds(const ble::BridgeState& state, const std::vector<ble::BridgeState>& worlds) const;

    [[nodiscard]] ParetoFront Search(const ble::BridgeState& state) const;

  private:
    std::shared_ptr<Resampler> resampler_;
    const AlphaMuConfig cfg_;
};

class AlphaMuBot final : public PlayBot {
  public:
    AlphaMuBot(const std::shared_ptr<Resampler>& resampler, const AlphaMuConfig cfg) : tt_(), resampler_(resampler),
      cfg_(cfg) {
    }

    AlphaMuBot(const std::shared_ptr<Resampler>& resampler,
               const AlphaMuConfig cfg,
               ble::Player player_id) : tt_(), resampler_(resampler),
                                        cfg_(cfg), player_id_(player_id) {
    }

    ble::BridgeMove Step(const ble::BridgeState& state) override;

    [[nodiscard]] ParetoFront Search(const ble::BridgeStateWithoutHiddenInfo& state,
                                     int num_max_moves,
                                     const Worlds& worlds,
                                     const ParetoFront& alpha);

    const TranspositionTable& GetTT() const { return tt_; }

    void SetTT(const TranspositionTable& tt) { tt_ = tt; }

    std::string Name() const override {
      return absl::StrFormat("AlphaMu, %d worlds, %d max moves", cfg_.num_worlds, cfg_.num_max_moves);
    }

  private:
    TranspositionTable tt_;
    std::shared_ptr<Resampler> resampler_;
    const AlphaMuConfig cfg_;
    const ble::Player player_id_{};
    std::optional<ParetoFront> last_iteration_front_{};
};

std::unique_ptr<PlayBot> MakeAlphaMuBot(ble::Player player_id, AlphaMuConfig cfg);

#endif // BRIDGE_LEARNING_PLAYCC_ALPHA_MU_BOT_H_
