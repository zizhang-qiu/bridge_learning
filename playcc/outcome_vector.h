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

//bool operator==(const OutcomeVector &lhs, const OutcomeVector &rhs) {
//  return (lhs.game_status == rhs.game_status) && (lhs.possible_world == rhs.possible_world);
//}

bool OutcomeVectorDominate(const OutcomeVector &lhs, const OutcomeVector &rhs);

#endif //BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
