#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_GAME_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_GAME_H_
#include "bridge_card.h"
#include "bridge_move.h"
#include "bridge_utils.h"
#include "utils.h"

namespace bridge_learning_env {
class BridgeGame {
  public:
    // Acceptable parameters
    //
    // "is_dealer_vulnerable": Whether the dealer side is vulnerable. (default false)
    // "is_non_dealer_vulnerable": Whether the non-dealer side is vulnerable. (default false)
    // "dealer": The dealer of the game, first player to make a call in auction phase.
    // "seed": Pseudo-random number generator seed. (default -1)
    explicit BridgeGame(const GameParameters& params);

    int NumDistinctActions() const { return kNumCards + kNumCalls; }

    int MaxChanceOutcomes() const { return kNumCards; }

    // Four player passes.
    int MinGameLength() const { return MaxChanceOutcomes() + kNumPlayers; }

    int MaxGameLength() const {
      return kNumCards     // 52 cards for deal
             + kNumPlayers // Opening pass
             + 9 * kNumBids
             // In auction, each bid can lead a sequence like 1C-pass-pass-double-pass-pass-redouble-pass-pass
             + kNumCards; // 52 card for play
    }

    int MaxMoves() const { return MaxAuctionMoves() + MaxPlayMoves(); }

    int MaxUtility() const { return kMaxUtility; }

    int MinUtility() const { return kMinUtility; }

    GameParameters Parameters() const;

    std::string Name() const { return kGameName; }

    std::string ShortName() const { return kShortName; }

    int HandSize() const { return hand_size; }

    bool IsDealerVulnerable() const { return is_dealer_vulnerable_; }

    bool IsNonDealerVulnerable() const { return is_non_dealer_vulnerable_; }

    bool IsPlayerVulnerable(Player player) const;

    bool IsPartnershipVulnerable(int partnership) const;

    Player Dealer() const { return dealer_; }

    BridgeMove GetMove(int uid) const { return moves_[uid]; }

    BridgeMove GetChanceOutcome(int uid) const { return chance_outcomes_[uid]; }

    int GetMoveUid(BridgeMove move) const;

    int GetMoveUid(BridgeMove::Type move_type,
                   Suit suit,
                   int rank,
                   Denomination denomination,
                   int level,
                   OtherCalls other_call) const;

    int GetChanceOutComeUid(BridgeMove move) const;

    BridgeMove PickRandomChance(
        const std::pair<std::vector<BridgeMove>, std::vector<double>>&
        chance_outcomes) const;

    int Seed() const { return seed_; }

    std::string ToString() const;

  private:
    GameParameters params_;
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

const GameParameters default_game_params = {};
const std::shared_ptr<BridgeGame> default_game = std::make_shared<BridgeGame>(
    default_game_params);

} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_GAME_H_
