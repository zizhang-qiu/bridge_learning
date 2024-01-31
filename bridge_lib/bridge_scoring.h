#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_SCORING_H
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_SCORING_H

#include <array>
#include <iostream>

#include "bridge_utils.h"

namespace bridge_learning_env {

struct Contract {
  int level{0};
  Denomination denomination{kNoTrump};
  DoubleStatus double_status{kUndoubled};
  Player declarer{-1};

  std::string ToString() const;

  int Index() const;

  bool operator==(const Contract &rhs) const;
};

std::ostream &operator<<(std::ostream &stream, const Contract &contract);

int Score(Contract contract, int declarer_tricks, bool is_vulnerable);

// All possible contracts.
inline constexpr int kNumContracts = kNumBids * kNumPlayers * kNumDoubleStatus + 1;

constexpr std::array<Contract, kNumContracts> AllContracts() {
  std::array<Contract, kNumContracts> contracts;
  int i = 0;
  contracts[i++] = Contract();
  for (const int level : {1, 2, 3, 4, 5, 6, 7}) {
    for (const Denomination trumps : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
        for (const DoubleStatus double_status : {kUndoubled, kDoubled, kRedoubled}) {
          contracts[i++] = Contract{level, trumps, double_status, declarer};
        }
      }
    }
  }
  return contracts;
}

inline constexpr std::array<Contract, kNumContracts> kAllContracts = AllContracts();

int GetImp(int score1, int score2);
} // namespace bridge_learning_env

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_SCORING_H
