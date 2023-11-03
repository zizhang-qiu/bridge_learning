//
// Created by qzz on 2023/11/3.
//

#ifndef BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
#define BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
#include <vector>
#include "vector_utils.h"
struct OutcomeVector {
  std::vector<int> game_status;
  std::vector<bool> possible_world;
};

bool OutcomeVectorDominate(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  if (lhs.possible_world != rhs.possible_world) {
    return false;
  }
  CheckVectorSize(lhs.game_status, rhs.game_status);
  if (!VectorGreaterEqual(lhs.game_status, rhs.game_status)) {
    return false;
  }
  size_t size = lhs.game_status.size();
  for (size_t i = 0; i < size; ++i) {
    if (lhs.game_status[i] > rhs.game_status[i]) {
      return true;
    }
  }
  return false;
}
#endif //BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
