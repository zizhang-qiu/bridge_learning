//
// Created by qzz on 2023/11/5.
//
#include <algorithm>

#include "outcome_vector.h"
bool OutcomeVectorDominate(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  if (lhs.possible_world != rhs.possible_world) {
    return false;
  }
  CheckVectorSize(lhs.game_status, rhs.game_status);

  const size_t size = lhs.game_status.size();
  bool is_dominant = false;
  for (size_t i = 0; i < size; ++i) {
    if (!lhs.possible_world[i]) {
      continue;
    }
    if (lhs.game_status[i] < rhs.game_status[i]) {
      return false;
    } else if (lhs.game_status[i] > rhs.game_status[i]) {
      is_dominant = true;
    }
  }
  return is_dominant;
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
  return count == 0 ? 0.0 : sum / count;
}
std::string OutcomeVector::ToString() const {
  std::string rv;
  rv += "game_status:\n";
  rv += VectorToString(game_status);
  rv += "\npossible_worlds:\n";
  rv += VectorToString(possible_world);
  rv += "\nmove: ";
  rv += move.ToString();
  return rv;
}
bool operator==(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  return (lhs.game_status == rhs.game_status) && (lhs.possible_world == rhs.possible_world);
}

std::pair<int, bool> GetGameStatusAndPossibleWorlds(const int lhs_game_status,
                                                    const int rhs_game_status,
                                                    const bool lhs_possible,
                                                    const bool rhs_possible) {
  int game_status;
  bool possible_worlds;

  if (lhs_possible && rhs_possible) {
    return {std::min(lhs_game_status, rhs_game_status), true};
  } else {
    if((!lhs_possible) && (!rhs_possible)){
      // Both impossible
      return {0, false};
    }
    if (lhs_possible){
      return {lhs_game_status, true};
    }
    return {rhs_game_status, true};
  }
}

OutcomeVector OutcomeVectorJoin(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  const size_t size = lhs.game_status.size();
  std::vector<int> game_status(size, 0);
  std::vector<int> possible_worlds(size, 0);
  for (size_t i = 0; i < size; ++i) {
    const auto [status, possible] = GetGameStatusAndPossibleWorlds(
        lhs.game_status[i], rhs.game_status[i], lhs.possible_world[i], rhs.possible_world[i]);
    game_status[i] = status;
    possible_worlds[i] = possible;
  }
  return {game_status, possible_worlds};
}
