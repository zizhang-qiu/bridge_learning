//
// Created by qzz on 2023/11/5.
//
#include <algorithm>

#include "outcome_vector.h"
bool OutcomeVectorDominate(const OutcomeVector &lhs, const OutcomeVector &rhs) {
//    if (lhs.possible_world != rhs.possible_world) {
//      return false;
//    }
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
  if (lhs_possible == rhs_possible) {
    if (lhs_possible) {
      // Both possible.
      game_status = std::min(lhs_game_status, rhs_game_status);
      possible_worlds = true;
      return std::make_pair(game_status, possible_worlds);
    }
    // Both impossible
    game_status = 0;
    possible_worlds = false;
    return std::make_pair(game_status, possible_worlds);
  }

  // One is possible while another is not.
  if (lhs_possible) {
    // lhs true, and rhs false
    // lhs status | rhs status | status | possible
    //     0      |      1     |   1    |  false
    //     1      |      1     |   1    |  true
    //     1      |      0     |   1    |  true
    //     0      |      0     |   0    |  true
    game_status = std::max(lhs_game_status, rhs_game_status);
    possible_worlds = !(lhs_game_status == 0 && rhs_game_status == 1);
    return std::make_pair(game_status, possible_worlds);
  }

  // lhs false, and rhs true
  // lhs status | rhs status | status | possible
  //     0      |      1     |   1    |  true
  //     1      |      1     |   1    |  true
  //     1      |      0     |   1    |  false
  //     0      |      0     |   0    |  true
  game_status = std::max(lhs_game_status, rhs_game_status);
  possible_worlds = !(lhs_game_status == 1 && rhs_game_status == 0);
  return std::make_pair(game_status, possible_worlds);
}

OutcomeVector OutcomeVectorJoin(const OutcomeVector &lhs, const OutcomeVector &rhs) {
  const size_t size = lhs.game_status.size();
  std::vector<int> game_status(size, 0);
  std::vector<bool> possible_worlds(size, false);
  for (size_t i = 0; i < size; ++i) {
    const auto [status, possible] = GetGameStatusAndPossibleWorlds(
        lhs.game_status[i], rhs.game_status[i], lhs.possible_world[i], rhs.possible_world[i]);
    game_status[i] = status;
    possible_worlds[i] = possible;
  }
  return {game_status, possible_worlds};
}
