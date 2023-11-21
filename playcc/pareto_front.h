//
// Created by qzz on 2023/11/5.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PARETO_FRONT_H_
#define BRIDGE_LEARNING_PLAYCC_PARETO_FRONT_H_
#include "outcome_vector.h"
class ParetoFront {
  public:
  ParetoFront() = default;
  explicit ParetoFront(const std::vector<OutcomeVector>& outcome_vectors);

  [[nodiscard]] int Size() const { return static_cast<int>(outcome_vectors_.size()); }

  bool Insert(const OutcomeVector& outcome_vector);

  [[nodiscard]] std::string ToString() const;

  [[nodiscard]] std::vector<OutcomeVector> OutcomeVectors() const { return outcome_vectors_; }

  [[nodiscard]] bool Empty() const { return outcome_vectors_.empty(); }

  static ParetoFront ParetoFrontWithOneOutcomeVector(const std::vector<bool>& possible_worlds, int fill_value);

  private:
  std::vector<OutcomeVector> outcome_vectors_;

  [[nodiscard]] bool HasSameVector(const OutcomeVector& outcome_vector) const;
};

ParetoFront operator*(const ParetoFront& lhs, const ParetoFront& rhs);

bool operator==(const ParetoFront& lhs, const ParetoFront& rhs);

bool operator<=(const ParetoFront& lhs, const ParetoFront& rhs);

ParetoFront ParetoFrontMin(const ParetoFront& lhs, const ParetoFront& rhs);

ParetoFront ParetoFrontMax(const ParetoFront& lhs, const ParetoFront& rhs);
#endif // BRIDGE_LEARNING_PLAYCC_PARETO_FRONT_H_
