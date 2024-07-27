//
// Created by qzz on 2023/9/23.
//

#ifndef BRIDGE_LIB_BRIDGE_STATE_H
#define BRIDGE_LIB_BRIDGE_STATE_H
#include <random>
#include <iostream>
#include <set>

// #include "third_party/dds/src/Memory.h"
// #include "third_party/dds/src/SolverIF.h"
// #include "third_party/dds/src/TransTableL.h"
#include "third_party/dds/include/dll.h"

#include "auction_tracker.h"
#include "bridge_card.h"
#include "bridge_game.h"
#include "bridge_hand.h"
#include "bridge_history_item.h"
#include "bridge_utils.h"
#include "trick.h"
#include "bridge_deck.h"

namespace bridge_learning_env {
class BridgeState {
  public:
    ~BridgeState() = default;

    BridgeState(const BridgeState&) = default;

    explicit BridgeState(std::shared_ptr<BridgeGame> parent_game);

    Contract GetContract() const { return contract_; }

    const BridgeDeck& Deck() const { return deck_; }

    const std::vector<BridgeHand>& Hands() const { return hands_; }

    const std::vector<BridgeHistoryItem>& History() const {
      return move_history_;
    }

    std::vector<int> UidHistory() const;

    const std::vector<BridgeCard>& PlayedCards() const { return played_cards_; }

    Player CurrentPlayer() const;

    bool IsDummyActing() const;

    bool IsDummyCardShown() const {
      // After the opening lead is faced, dummy spreads his hand in front of him on the table, face up.
      return num_cards_played_ >= 1;
    }

    const BridgeHand& DummyHand() const {
      REQUIRE(num_cards_played_>=1);
      const auto dummy = GetDummy();
      return hands_[dummy];
    }

    bool IsInPhase(const Phase phase) const { return phase_ == phase; }

    Player GetDummy() const;

    void ApplyMove(const BridgeMove& move);

    void ApplyMoveWithLegalityCheck(const BridgeMove& move);

    bool MoveIsLegal(const BridgeMove& move) const;

    bool IsTerminal() const { return phase_ == Phase::kGameOver; }

    std::shared_ptr<BridgeGame> ParentGame() const { return parent_game_; }

    std::string ToString() const;

    bool operator==(const BridgeState& other) const {
      return ToString() == other.ToString();
    }

    double ChanceOutcomeProb(const BridgeMove& move) const;

    // Get the valid chance moves, and associated probabilities.
    // Guaranteed that moves.size() == probabilities.size().
    std::pair<std::vector<BridgeMove>, std::vector<double>>
    ChanceOutcomes() const;

    void ApplyRandomChance();

    Phase CurrentPhase() const { return phase_; }

    std::vector<BridgeMove> LegalMoves(Player player) const;

    std::vector<BridgeMove> LegalMoves() const;

    std::vector<int> ScoreForContracts(Player player,
                                       const std::vector<int>& contracts) const;

    std::array<std::array<int, kNumPlayers>, kNumDenominations>
    DoubleDummyResults(bool dds_order = false) const;

    void SetDoubleDummyResults(
        const std::vector<int>& double_dummy_tricks) const;

    void SetDoubleDummyResults(
        const std::array<int, kNumPlayers * kNumDenominations>&
        double_dummy_tricks) const;

    bool IsPlayerVulnerable(Player player) const;

    BridgeState Clone() const { return (*this); }

    int NumTricksPlayed() const { return num_cards_played_ / kNumPlayers; }

    Trick& CurrentTrick() { return tricks_[num_cards_played_ / kNumPlayers]; }

    const Trick& CurrentTrick() const {
      return tricks_[num_cards_played_ / kNumPlayers];
    }

    std::array<Trick, kNumTricks> Tricks() const { return tricks_; }

    int NumCardsPlayed() const { return num_cards_played_; }

    std::vector<BridgeHand> OriginalDeal() const;

    bool IsChanceNode() const { return CurrentPlayer() == kChancePlayerId; }

    std::vector<int> Scores() const { return scores_; }

    int NumDeclarerTricks() const { return num_declarer_tricks_; }

    BridgeState Child(const BridgeMove& move) const;

    std::string Serialize() const;

    static BridgeState Deserialize(const std::string& str,
                                   const std::shared_ptr<BridgeGame>& game);

    std::vector<BridgeHistoryItem> DealHistory() const {
      return SpecifiedHistory(BridgeMove::Type::kDeal);
    }

    std::vector<BridgeHistoryItem> PlayHistory() const {
      return SpecifiedHistory(BridgeMove::Type::kPlay);
    }

    std::vector<BridgeHistoryItem> AuctionHistory() const {
      return SpecifiedHistory(BridgeMove::Type::kAuction);
    }

    const BridgeMove& LastMove() const {
      REQUIRE(!move_history_.empty());
      return move_history_.back().move;
    }

  private:
    BridgeDeck deck_;
    Phase phase_;
    std::array<Trick, kNumTricks> tricks_;
    Player dealer_;
    Player current_player_;
    std::shared_ptr<BridgeGame> parent_game_ = nullptr;
    std::vector<BridgeHand> hands_;
    std::vector<BridgeHistoryItem> move_history_;
    std::vector<BridgeCard> played_cards_;
    AuctionTracker auction_tracker_;
    Contract contract_;
    Player last_round_winner_;
    int num_cards_played_;
    int num_declarer_tricks_;
    std::vector<int> scores_;
    bool is_dealer_vulnerable_;
    bool is_non_dealer_vulnerable_;
    mutable std::optional<ddTableResults> double_dummy_results_{};

    void AdvanceToNextPlayer();

    bool DealIsLegal(const BridgeMove& move) const;

    bool AuctionIsLegal(const BridgeMove& move) const;

    bool PlayIsLegal(const BridgeMove& move) const;

    std::string FormatVulnerability() const;

    std::string FormatDeal() const;

    std::string FormatAuction() const;

    std::string FormatPlay() const;

    std::string FormatResult() const;

    Player PlayerToDeal() const;

    void ScoreUp();

    void ComputeDoubleDummyTricks() const;

    std::vector<BridgeHistoryItem>
    SpecifiedHistory(BridgeMove::Type type) const;
};

std::ostream& operator<<(std::ostream& stream, const BridgeState& state);
} // namespace bridge_learning_env

#endif /* BRIDGE_LIB_BRIDGE_STATE_H */
