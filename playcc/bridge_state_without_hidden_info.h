//
// Created by qzz on 2023/11/28.
//

#ifndef BRIDGE_LEARNING_PLAYCC_BRIDGE_STATE_WITHOUT_HIDDEN_INFO_H_
#define BRIDGE_LEARNING_PLAYCC_BRIDGE_STATE_WITHOUT_HIDDEN_INFO_H_

#include "bridge_lib/auction_tracker.h"
#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_history_item.h"
#include "bridge_lib/bridge_move.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/trick.h"
namespace bridge_learning_env {

// The state used in \alpha\mu search.
// This state doesn't need dealing phase, because we don't want hidden information.
class BridgeStateWithoutHiddenInfo {
  public:
  BridgeStateWithoutHiddenInfo(const std::shared_ptr<BridgeGame> &parent_game, const BridgeHand &dummy_hand);

  BridgeStateWithoutHiddenInfo(const std::shared_ptr<BridgeGame> &parent_game,
                               const Contract &contract,
                               const BridgeHand &dummy_hand);

  explicit BridgeStateWithoutHiddenInfo(const BridgeState &state);

  BridgeStateWithoutHiddenInfo(const BridgeStateWithoutHiddenInfo &) = default;

  static BridgeStateWithoutHiddenInfo FromBridgeState(const BridgeState &state, bool with_bidding_history = true);
  [[nodiscard]] Player CurrentPlayer() const;
  [[nodiscard]] std::shared_ptr<BridgeGame> ParentGame() const { return parent_game_; }
  [[nodiscard]] int NumTricksPlayed() const { return num_cards_played_ / kNumPlayers; }
  Trick &CurrentTrick() { return tricks_[num_cards_played_ / kNumPlayers]; }
  [[nodiscard]] const Trick &CurrentTrick() const { return tricks_[num_cards_played_ / kNumPlayers]; }
  [[nodiscard]] Phase CurrentPhase() const { return phase_; }
  [[nodiscard]] bool MoveIsLegal(const BridgeMove &move) const;

  void ApplyMove(const BridgeMove &move);
  [[nodiscard]] bool IsTerminal() const { return phase_ == Phase::kGameOver; }
  [[nodiscard]] int NumCardsPlayed() const { return num_cards_played_; }
  [[nodiscard]] std::vector<int> Scores() const { return scores_; }
  [[nodiscard]] int NumDeclarerTricks() const { return num_declarer_tricks_; }
  [[nodiscard]] std::vector<BridgeMove> LegalMoves(Player player) const;
  [[nodiscard]] std::vector<BridgeMove> LegalMoves() const { return LegalMoves(current_player_); }
  [[nodiscard]] BridgeStateWithoutHiddenInfo Child(const BridgeMove &move) const;
  [[nodiscard]] std::vector<BridgeHistoryItem> PlayHistory() const;
  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] std::array<bool, kNumSuits> VoidSuitsForPlayer(Player player) const;
  [[nodiscard]] Player GetDummy() const;
  [[nodiscard]] Contract GetContract() const { return contract_; }

  void SetDummyHand(const BridgeHand &dummy_hand);

  private:
  std::shared_ptr<BridgeGame> parent_game_ = nullptr;
  std::vector<BridgeHistoryItem> move_history_;
  AuctionTracker auction_tracker_;
  Contract contract_;
  Phase phase_;
  std::array<Trick, kNumTricks> tricks_;
  Player dealer_;
  Player current_player_;
  Player last_round_winner_;
  bool is_dealer_vulnerable_;
  bool is_non_dealer_vulnerable_;
  int num_cards_played_;
  int num_declarer_tricks_;
  std::vector<int> scores_;
  std::optional<BridgeHand> dummy_hand_;

  void AdvanceToNextPlayer();
  [[nodiscard]] bool AuctionIsLegal(const BridgeMove &move) const;
  [[nodiscard]] bool PlayIsLegal(const BridgeMove &move) const;

  [[nodiscard]] std::string FormatVulnerability() const;
  [[nodiscard]] std::string FormatDeal() const;
  [[nodiscard]] std::string FormatAuction() const;
  [[nodiscard]] std::string FormatPlay() const;
  [[nodiscard]] std::string FormatResult() const;

  void ScoreUp();
};

} // namespace bridge_learning_env

#endif // BRIDGE_LEARNING_PLAYCC_BRIDGE_STATE_WITHOUT_HIDDEN_INFO_H_
