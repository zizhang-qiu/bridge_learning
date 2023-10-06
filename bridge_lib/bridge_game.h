#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_GAME_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_GAME_H_
#include "bridge_card.h"
#include "bridge_move.h"
#include "bridge_state.h"
#include "bridge_utils.h"
#include "utils.h"

namespace bridge {
class BridgeGame {
 public:
  // Acceptable parameters
  //
  // "is_dealer_vulnerable": Whether the dealer side is vulnerable. (default false)
  // "is_non_dealer_vulnerable": Whether the non-dealer side is vulnerable. (default false)
  // "dealer": The dealer of the game, first player to make a call in auction phase.
  // "seed": Pseudo-random number generator seed. (default -1)
  explicit BridgeGame(const GameParameters& params);
  std::shared_ptr<BridgeState> NewInitialState() const;
  int NumDistinctActions() const { return kNumCards + kNumCalls; }
  int MaxChanceOutcomes() const { return kNumCards; }
  int MaxMoves() const;
  int MaxUtility() const { return kMaxUtility; }
  int MinUtility() const { return kMinUtility; }
  bridge::GameParameters Parameters() const;
  std::string Name() const{return "Contract Bridge";}
  std::string ShortName() const{return "bridge";}

  int HandSize() const{return hand_size;}

  bool IsDealerVulnerable() const { return is_dealer_vulnerable_; }
  bool IsNonDealerVulnerable() const { return is_non_dealer_vulnerable_; }
  bool IsPlayerVulnerable(Player player) const;
  bool IsPartnershipVulnerable(int partnership) const;
  Player Dealer() const{return dealer_;}

  BridgeMove GetMove(int uid) const { return moves_[uid]; }
  BridgeMove GetChanceOutcome(int uid) const { return chance_outcomes_[uid]; }
  int GetMoveUid(BridgeMove move) const;
  int GetMoveUid(BridgeMove::Type move_type,
                 Suit suit,
                 int rank,
                 Denomination denomination,
                 int level,
                 OtherCalls other_call) const;

  BridgeMove PickRandomChance(const std::pair<std::vector<BridgeMove>, std::vector<double>> &chance_outcomes) const;

 private:
  bridge::GameParameters params_;
  bool is_dealer_vulnerable_;
  bool is_non_dealer_vulnerable_;
  Player dealer_;
  int MaxAuctionMoves() const { return kNumCalls; }
  int MaxPlayMoves() const { return kNumCards; }
  const int hand_size = kNumCardsPerHand;
  BridgeMove ConstructMove(int uid) const;
  BridgeMove ConstructChanceMove(int uid) const;

  // Table of all possible moves in this game.
  std::vector<BridgeMove> moves_;
  // Table of all possible chance outcomes in this game.
  std::vector<BridgeMove> chance_outcomes_;
  int seed_;
  mutable std::mt19937 rng_;
};
} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_GAME_H_
