//
// Created by qzz on 2023/11/5.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PARETO_FRONT_H_
#define BRIDGE_LEARNING_PLAYCC_PARETO_FRONT_H_
#include <ostream>

#include "outcome_vector.h"
class ParetoFront {
 public:
  ParetoFront() = default;
  explicit ParetoFront(const std::vector<OutcomeVector> &outcome_vectors);

  ParetoFront(const ParetoFront &) = default;

  [[nodiscard]] int Size() const { return static_cast<int>(outcome_vectors_.size()); }

  bool Insert(const OutcomeVector &outcome_vector);

  [[nodiscard]] std::string ToString() const;

  [[nodiscard]] std::vector<OutcomeVector> OutcomeVectors() const { return outcome_vectors_; }

  [[nodiscard]] bool Empty() const { return outcome_vectors_.empty(); }

  [[nodiscard]] double Score() const;

  [[nodiscard]] OutcomeVector BestOutcome() const;

  void SetMove(const ble::BridgeMove &move);

  static ParetoFront ParetoFrontWithOneOutcomeVector(const std::vector<int> &possible_worlds, int fill_value);

  std::string Serialize() const;

  static ParetoFront Deserialize(const std::string &str);

 private:
  std::vector<OutcomeVector> outcome_vectors_;

  [[nodiscard]] bool HasSameVector(const OutcomeVector &outcome_vector) const;
};

ParetoFront operator*(const ParetoFront &lhs, const ParetoFront &rhs);

bool operator==(const ParetoFront &lhs, const ParetoFront &rhs);

bool operator<=(const ParetoFront &lhs, const ParetoFront &rhs);

ParetoFront ParetoFrontMin(const ParetoFront &lhs, const ParetoFront &rhs);

ParetoFront ParetoFrontMax(const ParetoFront &lhs, const ParetoFront &rhs);

bool ParetoFrontDominate(const ParetoFront &lhs, const ParetoFront &rhs);

std::ostream &operator<<(std::ostream &stream, const ParetoFront &front);
//struct BridgeMoveOutcomeVector{
//  ble::BridgeMove move;
//  OutcomeVector outcome_vector;
//};
#endif // BRIDGE_LEARNING_PLAYCC_PARETO_FRONT_H_
