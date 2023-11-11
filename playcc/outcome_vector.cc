//
// Created by qzz on 2023/11/5.
//
#include "outcome_vector.h"
#include <algorithm>
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
bool operator==(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  return (lhs.game_status == rhs.game_status) && (lhs.possible_world == rhs.possible_world);
}
bool operator<(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  return std::lexicographical_compare(lhs.game_status.begin(), lhs.game_status.end(),
                                      rhs.game_status.begin(), rhs.game_status.end());
}
bool operator>(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  return std::lexicographical_compare(rhs.game_status.begin(), rhs.game_status.end(),
                                      lhs.game_status.begin(), lhs.game_status.end());
}
