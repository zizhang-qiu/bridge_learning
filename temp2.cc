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
#include "playcc/file.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"
#include "playcc/play_bot.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

class AlphaMuFactory : public BotFactory {
  public:
  ~AlphaMuFactory() = default;

  std::unique_ptr<PlayBot> Create(std::shared_ptr<const ble::BridgeGame> game,
                                  ble::Player player,
                                  const ble::GameParameters&bot_params) override {
    int num_max_moves = ble::ParameterValue<int>(bot_params, "num_max_moves", 1);
    int num_worlds = ble::ParameterValue<int>(bot_params, "num_worlds", 20);
    bool early_cut = ble::ParameterValue<bool>(bot_params, "early_cut", false);
    bool root_cut = ble::ParameterValue<bool>(bot_params, "root_cut", false);
    bool use_tt = ble::ParameterValue<bool>(bot_params, "use_tt", false);
    AlphaMuConfig cfg{num_max_moves, num_worlds, false, use_tt, early_cut, root_cut};
    return MakeAlphaMuBot(player, cfg);
  }
};

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
