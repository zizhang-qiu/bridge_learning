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
  const size_t size = lhs.game_status.size();
  for (size_t i = 0; i < size; ++i) {
    if (lhs.game_status[i] > rhs.game_status[i]) {
      return true;
    }
  }
  return false;
}
double OutcomeVector::Score() const {
  double sum = 0;
  int count = 0;
  for (size_t i = 0; i < game_status.size(); ++i) {
    if (possible_world[i]) {
      sum += game_status[i];
      count += 1;
    }
  }
  return sum / count;
}
bool operator==(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  return (lhs.game_status == rhs.game_status) && (lhs.possible_world == rhs.possible_world);
}
