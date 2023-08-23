#ifndef BRIDGE_SCORING
#define BRIDGE_SCORING
#include <array>
#include "bridge_utils.h"
namespace bridge {

struct Contract {
  int level{0};
  Denomination denomination{kNoTrump};
  DoubleStatus double_status{kUndoubled};
  Player declarer{-1};
  std::string ToString() const;
  int Index() const;
};

int Score(Contract contract, int declarer_tricks, bool is_vulnerable);

// All possible contracts.
inline constexpr int kNumContracts =
    kNumBids * kNumPlayers * kNumDoubleStates + 1;
constexpr std::array<Contract, kNumContracts> AllContracts() {
  std::array<Contract, kNumContracts> contracts;
  int i = 0;
  contracts[i++] = Contract();
  for (int level : {1, 2, 3, 4, 5, 6, 7}) {
    for (Denomination trumps :
         {kClubs, kDiamonds, kHearts, kSpades, kNoTrump}) {
      for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
        for (DoubleStatus double_status : {kUndoubled, kDoubled, kRedoubled}) {
          contracts[i++] = Contract{level, trumps, double_status, declarer};
        }
      }
    }
  }
  return contracts;
}
inline constexpr std::array<Contract, kNumContracts> kAllContracts =
    AllContracts();
}  // namespace bridge

#endif /* BRIDGE_SCORING */
