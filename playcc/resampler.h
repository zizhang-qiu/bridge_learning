//
// Created by qzz on 2023/10/22.
//

#ifndef PLAYCC_RESAMPLER_H
#define PLAYCC_RESAMPLER_H
#include <algorithm>
#include <array>

#include "bridge_lib/bridge_state.h"
#include "bridge_lib/utils.h"
#include "rela/utils.h"

#include "deck_sampler.h"
#include "utils.h"
namespace ble = bridge_learning_env;

struct ResampleResult {
  // For some reason, the resample procedure may fail.
  bool success;
  std::array<int, ble::kNumCards> result;
};

class Resampler {
  public:
    Resampler() = default;

    virtual ~Resampler() = default;

    virtual ResampleResult Resample(const ble::BridgeState& state) = 0;

    virtual void ResetWithParams(const std::unordered_map<std::string, std::string>&) {}
};

std::vector<std::array<int, ble::kNumCards>> ResampleMultipleDeals(const std::shared_ptr<Resampler>& resampler,
                                                                   const ble::BridgeState& state,
                                                                   int num_deals);

const std::vector<int> all_cards = ble::Arange(0, ble::kNumCards);

// Constraints for resampling, tracks for each player and each suit, how many cards at most can the player have.
// In play history, if a player cannot follow a suit, it means he has run out of this suit
// Thus a constraint should be considered to avoid breaking rules.
using ResampleConstraints = std::array<std::array<int, ble::kNumSuits>, ble::kNumPlayers>;

std::tuple<ResampleConstraints, std::vector<ble::BridgeHand>> GetKnownCardsAndConstraintsFromState(
  const ble::BridgeState& state);

class UniformResampler final : public Resampler {
  public:
    explicit UniformResampler(const int seed) : rng_(seed), deck_sampler_() {
    }

    ResampleResult Resample(const ble::BridgeState& state) override;

    void ResetWithParams(const std::unordered_map<std::string, std::string>& params) override;

  private:
    std::mt19937 rng_;
    DeckSampler deck_sampler_;
};
#endif /* PLAYCC_RESAMPLER_H */
