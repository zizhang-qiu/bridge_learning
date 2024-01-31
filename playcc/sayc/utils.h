//
// Created by qzz on 2024/1/30.
//

#ifndef BRIDGE_LEARNING_PLAYCC_SYAC_UTILS_H_
#define BRIDGE_LEARNING_PLAYCC_SYAC_UTILS_H_
#include "bridge_lib/bridge_observation.h"

namespace ble = bridge_learning_env;

namespace sayc {
bool HasOpeningBidBeenMade(const ble::BridgeObservation& obs);
}
#endif //BRIDGE_LEARNING_PLAYCC_SYAC_UTILS_H_
