#include "bridge_game.h"

namespace bridge {

BridgeGame::BridgeGame(const Parameters params) : params_(params) {}
bool BridgeGame::IsDealerVulnerable() const {
    return ParameterValue(params_, "is_dealer_vulnerable", false);
}

bool BridgeGame::IsNonDealerVulnerable() const{
    return ParameterValue(params_, "is_non_dealer_vulnerable", false);
}
}