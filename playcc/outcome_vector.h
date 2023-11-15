//
// Created by qzz on 2023/11/3.
//

#ifndef BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
#define BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
#include <vector>
#include "vector_utils.h"
struct OutcomeVector {

  [[nodiscard]] double Score() const;

  std::vector<int> game_status;
  std::vector<bool> possible_world;
};

bool operator==(const OutcomeVector &lhs, const OutcomeVector &rhs);

// These operators are just for sorting.
bool operator<(const OutcomeVector &lhs, const OutcomeVector &rhs);

bool operator>(const OutcomeVector &lhs, const OutcomeVector &rhs);

bool OutcomeVectorDominate(const OutcomeVector &lhs, const OutcomeVector &rhs);

#endif //BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
