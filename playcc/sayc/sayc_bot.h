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

// A bot make calls confirming to SAYC bidding system.
class SAYCBot {
  public:
    ble::BridgeMove Step(const ble::BridgeObservation& obs);

  private:
    // Belief stores information of other three players.
    // Start from next player, i.e. Left hand opponent.
    std::array<HandInfo, ble::kNumPlayers - 1> belief_;

    // Make the hand analyzer to const since our hand will not change.
    const HandAnalyzer hand_analyzer_;

    ble::BridgeMove NoTrumpOpening(const ble::BridgeObservation& obs) const;
};

}
#endif //SAYC_BOT_H