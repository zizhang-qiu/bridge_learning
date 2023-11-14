//
// Created by qzz on 2023/11/5.
//
#include "pareto_front.h"
#include <algorithm>
ParetoFront::ParetoFront(const std::vector<OutcomeVector> &outcome_vectors) {
  if (outcome_vectors.empty()) {
    outcome_vectors_ = outcome_vectors;
  }
  size_t size = outcome_vectors.size();
  // For each outcome vector, if another vector dominates it, it should be eliminated.
  for (size_t i = 0; i < size; ++i) {
    bool is_dominated = false;
    bool has_same = false;
    for (size_t j = 0; j < size; ++j) {
      if (OutcomeVectorDominate(outcome_vectors[j], outcome_vectors[i])) {
        is_dominated = true;
        break;
      }
    }
    for (const auto &ov : outcome_vectors_) {
      if (ov == outcome_vectors[i]) {
        has_same = true;
        break;
      }
    }
    if (!is_dominated && !has_same) {
      outcome_vectors_.push_back(outcome_vectors[i]);
    }
  }
}

bool ParetoFront::Insert(const OutcomeVector &outcome_vector) {
  if (outcome_vectors_.empty()) {
    outcome_vectors_.push_back(outcome_vector);
    return true;
  }

  for (auto &ov : outcome_vectors_) {
    if (OutcomeVectorDominate(ov, outcome_vector)) {
      return false;
    }
  }

  size_t original_size = outcome_vectors_.size();

  outcome_vectors_.erase(
      std::remove_if(outcome_vectors_.begin(),
                     outcome_vectors_.end(),
                     [outcome_vector](const OutcomeVector &vec) { return OutcomeVectorDominate(outcome_vector, vec); }),
      outcome_vectors_.end());

  bool is_one_dominated = outcome_vectors_.size() < original_size;
  if (is_one_dominated) {
    outcome_vectors_.push_back(outcome_vector);
    return true;
  }

  bool has_same_vector = HasSameVector(outcome_vector);
  if (!has_same_vector) {
    outcome_vectors_.push_back(outcome_vector);
    return true;
  }
  return false;
}

std::string ParetoFront::ToString() const {
  std::string rv = "Game status:\n{";
  for (auto it = outcome_vectors_.begin(); it != outcome_vectors_.end(); ++it) {
    rv += "[" + VectorToString((*it).game_status) + "]";
    if (std::next(it) != outcome_vectors_.end()) {
      rv += ", ";
    }
  }
  rv += "}\npossible worlds:\n";
  if (!outcome_vectors_.empty()) {
    rv += "[" + VectorToString(outcome_vectors_[0].possible_world) + "]";
  }
  return rv;
}
bool ParetoFront::HasSameVector(const OutcomeVector &outcome_vector) const {
  for (const auto &ov : outcome_vectors_) {
    if (ov == outcome_vector) {
      return true;
    }
  }
  return false;
}

ParetoFront operator*(const ParetoFront &lhs, const ParetoFront &rhs) {
  ParetoFront res{};
  for (const auto &ov_l : lhs.OutcomeVectors()) {
    for (const auto &ov_r : rhs.OutcomeVectors()) {
      auto product = VectorProduct(ov_l.game_status, ov_r.game_status);
      std::cout << VectorToString(product) << std::endl;
      const OutcomeVector outcome_vector{product, ov_l.possible_world};
      res.Insert(outcome_vector);
    }
  }
  return res;
}
bool operator==(const ParetoFront &lhs, const ParetoFront &rhs) { return lhs.OutcomeVectors() == rhs.OutcomeVectors(); }

bool operator<=(const ParetoFront &lhs, const ParetoFront &rhs) {
  for (const auto &vec : lhs.OutcomeVectors()) {
    bool one_greater_or_equal = false;
    for (const auto &v : rhs.OutcomeVectors()) {
      if (VectorGreaterEqual(v.game_status, vec.game_status)) {
        one_greater_or_equal = true;
        break;
      }
    }
    if (!one_greater_or_equal) {
      return false;
    }
  }
  return true;
}

ParetoFront ParetoFrontJoin(const ParetoFront &lhs, const ParetoFront &rhs) {
  ParetoFront result{};
  for (const auto &vec : lhs.OutcomeVectors()) {
    for (const auto &v : rhs.OutcomeVectors()) {
      const auto r_vec = VectorMin(vec.game_status, v.game_status);
      const OutcomeVector outcome_vector{r_vec, vec.possible_world};
      result.Insert(outcome_vector);
    }
  }
  return result;
}
