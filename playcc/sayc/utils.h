//
// Created by qzz on 2024/1/30.
//

#ifndef BRIDGE_LEARNING_PLAYCC_SYAC_UTILS_H_
#define BRIDGE_LEARNING_PLAYCC_SYAC_UTILS_H_
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "bridge_lib/bridge_observation.h"
#include "../common_utils/log_utils.h"

namespace ble = bridge_learning_env;

namespace sayc {
bool HasOpeningBidBeenMade(const ble::BridgeObservation& obs);

ble::BridgeMove ConstructBidMove(ble::Suit suit, int level);

ble::BridgeMove
ConstructAuctionMoveFromString(const std::string& call_string);


}
#endif //BRIDGE_LEARNING_PLAYCC_SYAC_UTILS_H_