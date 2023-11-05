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

  [[nodiscard]] int Size() const{return static_cast<int>(outcome_vectors_.size());}

  bool Insert(const OutcomeVector& outcome_vector);

  [[nodiscard]] std::string ToString() const;
 private:
  std::vector<OutcomeVector> outcome_vectors_;

};
#endif //BRIDGE_LEARNING_PLAYCC_PARETO_FRONT_H_
