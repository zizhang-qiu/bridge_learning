//
// Created by qzz on 2023/9/23.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_H_
#include <random>
#include <set>
#include "bridge_game.h"
#include "bridge_utils.h"
#include "bridge_card.h"
#include "bridge_hand.h"
#include "bridge_history_item.h"
#include "auction_tracker.h"
#include "trick.h"

#include "third_party/dds/src/Memory.h"
#include "third_party/dds/src/SolverIF.h"
#include "third_party/dds/src/TransTableL.h"
#include "bridge_deck.h"
namespace bridge_learning_env {
class BridgeState {
 public:

  explicit BridgeState(std::shared_ptr<BridgeGame> parent_game);

  Contract GetContract() const { return contract_; }

  const BridgeDeck &Deck() const { return deck_; }

  const std::vector<BridgeHand> &Hands() const { return hands_; }

  const std::vector<BridgeHistoryItem> &History() const { return move_history_; }

  std::vector<int> UidHistory() const;

  const std::vector<BridgeCard> &PlayedCard() const { return played_cards_; }

  Player CurrentPlayer() const;

  bool IsDummyActing() const;

  bool IsDummyCardShown() const {
    // After the opening lead is faced, dummy spreads his hand in front of him on the table, face up.
    return num_cards_played_ >= 1;
  }

  Player GetDummy() const;

  void ApplyMove(const BridgeMove &move);

  bool MoveIsLegal(const BridgeMove &move) const;

  bool IsTerminal() const { return phase_ == Phase::kGameOver; }

  std::shared_ptr<BridgeGame> ParentGame() const { return parent_game_; }

  std::string ToString() const;

  double ChanceOutcomeProb(const BridgeMove &move) const;

  // Get the valid chance moves, and associated probabilities.
  // Guaranteed that moves.size() == probabilities.size().
  std::pair<std::vector<BridgeMove>, std::vector<double>> ChanceOutcomes() const;

  void ApplyRandomChance();

  Phase CurrentPhase() const { return phase_; }

  std::vector<BridgeMove> LegalMoves(Player player) const;

  std::vector<BridgeMove> LegalMoves() const;

  std::vector<int> ScoreForContracts(Player player,
                                     const std::vector<int> &contracts) const;

  std::array<std::array<int, kNumPlayers>, kNumDenominations> DoubleDummyResults(bool dds_order = false) const;

  void SetDoubleDummyResults(const std::vector<int> &double_dummy_tricks) const;

  void SetDoubleDummyResults(const std::array<int, kNumPlayers * kNumDenominations> &double_dummy_tricks) const;

  bool IsPlayerVulnerable(Player player) const;

  std::unique_ptr<BridgeState> Clone() const {
    return std::make_unique<BridgeState>(*this);
  }

  int NumTricksPlayed() const { return num_cards_played_ / kNumPlayers; }

  Trick &CurrentTrick() { return tricks_[num_cards_played_ / kNumPlayers]; }
  const Trick &CurrentTrick() const {
    return tricks_[num_cards_played_ / kNumPlayers];
  }

  int NumCardsPlayed() const { return num_cards_played_; }

  std::vector<BridgeHand> OriginalDeal() const;

  bool IsChanceNode() const { return CurrentPlayer() == kChancePlayerId; }

  std::vector<int> Scores() const { return scores_; }

  int NumDeclarerTricks() const{return num_declarer_tricks_;}

  std::unique_ptr<BridgeState> Child(const BridgeMove& move) const;

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
  bool DealIsLegal(const BridgeMove &move) const;
  bool AuctionIsLegal(const BridgeMove &move) const;
  bool PlayIsLegal(const BridgeMove &move) const;

  std::string FormatVulnerability() const;
  std::string FormatDeal() const;
  std::string FormatAuction() const;
  std::string FormatPlay() const;
  std::string FormatResult() const;

  Player PlayerToDeal() const;
  void ScoreUp();
  void ComputeDoubleDummyTricks() const;

};
}

#endif //BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_H_
