//
// Created by qzz on 2023/12/22.
//
#include <iostream>
#include <algorithm>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"

#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "bridge_lib/bridge_state.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"
#include "playcc/play_bot.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

// REGISTER_PLAY_BOT("alpha_mu", AlphaMuFactory);

int main() {
  for(const auto& bot_name:RegisteredBots()) {
    std::cout << bot_name << std::endl;
  }

  // std::cout << RegisteredBots().size() << std::endl;
  //
  // std::cout << __COUNTER__ << std::endl;
  auto bot = LoadBot("alpha_mu", game, 0);

  return 0;
}
