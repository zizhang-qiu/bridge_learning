//
// Created by qzz on 2023/11/5.
//
#include "pareto_front.h"
ParetoFront::ParetoFront(const std::vector<OutcomeVector> &outcome_vectors) {
  if (outcome_vectors.empty()) {
    outcome_vectors_ = outcome_vectors;
  }
  size_t size = outcome_vectors.size();
  // For each outcome vector, if another vector dominates it, it should be eliminated.
  for (size_t i = 0; i < size; ++i) {
    bool is_dominated = false;
    for (size_t j = 0; j < size; ++j) {
      if (OutcomeVectorDominate(outcome_vectors[j], outcome_vectors[i])) {
        is_dominated = true;
        break;
      }
    }
    if (!is_dominated) {
      outcome_vectors_.push_back(outcome_vectors[i]);
    }
  }
}

bool ParetoFront::Insert(const OutcomeVector &outcome_vector) {
  if (outcome_vectors_.empty()) {
    outcome_vectors_.push_back(outcome_vector);
    return true;
  }

  for (const auto &ov : outcome_vectors_) {
    if (OutcomeVectorDominate(ov, outcome_vector)) {
      return false;
    }
  }

  outcome_vectors_.push_back(outcome_vector);
  return true;
}

std::string ParetoFront::ToString() const {
  std::string rv = "Game status:\n{";
  for (const auto &ov : outcome_vectors_) {
    rv += "[" + VectorToString(ov.game_status) + "]";
  }
  rv += "}\npossible worlds:\n";
  if (!outcome_vectors_.empty()) {
    rv += "[" + VectorToString(outcome_vectors_[0].possible_world) + "]";
  }
  return rv;
}
