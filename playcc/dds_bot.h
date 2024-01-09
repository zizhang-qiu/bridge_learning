//
// Created by qzz on 2023/10/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_CHEAT_BOT_H_
#define BRIDGE_LEARNING_PLAYCC_CHEAT_BOT_H_
#include "bridge_lib/bridge_state.h"
#include "play_bot.h"
#include "utils.h"

namespace ble = bridge_learning_env;
class DDSBot final : public PlayBot {
  public:
  DDSBot() { SetMaxThreads(0); }

  ble::BridgeMove Step(const ble::BridgeState &state) override;

  std::string Name() const override { return "DDS"; }
};

std::unique_ptr<PlayBot> MakeDDSBot(ble::Player player_id);
#endif // BRIDGE_LEARNING_PLAYCC_CHEAT_BOT_H_
