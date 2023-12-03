//
// Created by qzz on 2023/11/3.
//

#ifndef BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
#define BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
#include <vector>
#include "vector_utils.h"
#include "bridge_lib/bridge_move.h"
namespace ble = bridge_learning_env;
struct OutcomeVector {

  OutcomeVector(const OutcomeVector &) = default;

  [[nodiscard]] double Score() const;

  std::string ToString() const;

  std::vector<int> game_status;
  std::vector<bool> possible_world;
  ble::BridgeMove move{};

};

bool operator==(const OutcomeVector &lhs, const OutcomeVector &rhs);

std::ostream& operator<<(std::ostream& stream, OutcomeVector& outcome_vector);

bool OutcomeVectorDominate(const OutcomeVector &lhs, const OutcomeVector &rhs);

#endif //BRIDGE_LEARNING_PLAYCC_OUTCOME_VECTOR_H_
