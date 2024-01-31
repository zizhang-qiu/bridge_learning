//
// Created by qzz on 2024/1/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_SAYC_HAND_INFO_H_
#define BRIDGE_LEARNING_PLAYCC_SAYC_HAND_INFO_H_
#include "hand_analyzer.h"
#include "range.h"


namespace ble = bridge_learning_env;

namespace sayc {
// Information of a hand in other player's view.
struct HandInfo {
  Range HCP_range_;
  bool is_balanced = false;
  std::array<Range, ble::kNumSuits> suit_length_range_{};



};
}


#endif //BRIDGE_LEARNING_PLAYCC_SAYC_HAND_INFO_H_
