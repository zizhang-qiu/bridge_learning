//
// Created by qzz on 2023/10/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PLAY_BOT_H_
#define BRIDGE_LEARNING_PLAYCC_PLAY_BOT_H_
#include "bridge_lib/bridge_state_2.h"
namespace ble=bridge_learning_env;
class PlayBot{
 public:
  PlayBot()=default;

  virtual ~PlayBot()=default;

  virtual ble::BridgeMove Act(const ble::BridgeState2 &state) = 0;

};
#endif //BRIDGE_LEARNING_PLAYCC_PLAY_BOT_H_
