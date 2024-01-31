//
// Created by qzz on 24-1-31.
//

#ifndef SAYC_BOT_H
#define SAYC_BOT_H
#include "constraints.h"
#include "hand_info.h"
#include "hand_analyzer.h"

namespace ble = bridge_learning_env;

namespace sayc {
inline constexpr int kMinLengthForMajorOpening = 5;
inline constexpr std::array<int, ble::kNumSuits> kOneDiamondSpecialCase =
    {4, 4, 3, 2};

// A bot make calls confirming to SAYC bidding system.
class SAYCBot {
  public:
    SAYCBot() = default;

    ble::BridgeMove Step(const ble::BridgeObservation& obs);

    void InformObservation(const ble::BridgeObservation& obs);

    void Restart();

  private:
    // Belief stores information of other three players.
    // Start from next player, i.e. Left hand opponent.
    std::array<HandInfo, ble::kNumPlayers - 1> belief_;

    HandAnalyzer hand_analyzer_;

    std::vector<ble::BridgeHistoryItem> internal_history_;

    mutable std::mt19937 rng_;

    int GetSeat(const ble::BridgeObservation& obs) const;

    ble::BridgeMove NoTrumpOpening(const ble::BridgeObservation& obs) const;

    ble::BridgeMove OneLevelOpening(const ble::BridgeObservation& obs) const;

    ble::BridgeMove OneLevelOpeningImpl(
      const ble::BridgeObservation& obs) const;
};
}
#endif //SAYC_BOT_H
