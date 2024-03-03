//
// Created by qzz on 2023/11/28.
//

#ifndef PLAYCC_BRIDGE_STATE_WITHOUT_HIDDEN_INFO_H
#define PLAYCC_BRIDGE_STATE_WITHOUT_HIDDEN_INFO_H

#include "bridge_lib/auction_tracker.h"
#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_history_item.h"
#include "bridge_lib/bridge_move.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/trick.h"

namespace bridge_learning_env {
bool operator==(const BridgeHand& lhs, const BridgeHand& rhs);

// The state used in \alpha\mu search.
// This state doesn't need dealing phase, because we don't want hidden information.
class BridgeStateWithoutHiddenInfo {
 public:
  BridgeStateWithoutHiddenInfo(const std::shared_ptr<BridgeGame>& parent_game,
                               const BridgeHand& dummy_hand);

  BridgeStateWithoutHiddenInfo(const std::shared_ptr<BridgeGame>& parent_game,
                               const Contract& contract,
                               const BridgeHand& dummy_hand);

  BridgeStateWithoutHiddenInfo();

  explicit BridgeStateWithoutHiddenInfo(const BridgeState& state);

  BridgeStateWithoutHiddenInfo(const BridgeStateWithoutHiddenInfo&) = default;

  bool operator==(const BridgeStateWithoutHiddenInfo& other) const {
    return UidHistory() == other.UidHistory() &&
           ParentGame() == other.ParentGame();
  }

  static BridgeStateWithoutHiddenInfo FromBridgeState(
      const BridgeState& state, bool with_bidding_history = true);
  [[nodiscard]] Player CurrentPlayer() const;

  [[nodiscard]] std::shared_ptr<BridgeGame> ParentGame() const {
    return parent_game_;
  }

  [[nodiscard]] int NumTricksPlayed() const {
    return num_cards_played_ / kNumPlayers;
  }

  Trick& CurrentTrick() { return tricks_[num_cards_played_ / kNumPlayers]; }

  [[nodiscard]] const Trick& CurrentTrick() const {
    return tricks_[num_cards_played_ / kNumPlayers];
  }

  [[nodiscard]] Phase CurrentPhase() const { return phase_; }

  [[nodiscard]] bool MoveIsLegal(const BridgeMove& move) const;

  void ApplyMove(const BridgeMove& move);

  [[nodiscard]] bool IsTerminal() const { return phase_ == Phase::kGameOver; }

  [[nodiscard]] int NumCardsPlayed() const { return num_cards_played_; }

  [[nodiscard]] std::vector<int> Scores() const { return scores_; }

  [[nodiscard]] int NumDeclarerTricks() const { return num_declarer_tricks_; }

  [[nodiscard]] std::vector<BridgeMove> LegalMoves(Player player) const;

  [[nodiscard]] std::vector<BridgeMove> LegalMoves() const {
    return LegalMoves(current_player_);
  }

  [[nodiscard]] BridgeStateWithoutHiddenInfo Child(
      const BridgeMove& move) const;
  [[nodiscard]] std::vector<BridgeHistoryItem> PlayHistory() const;
  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] std::array<bool, kNumSuits> VoidSuitsForPlayer(
      Player player) const;
  [[nodiscard]] Player GetDummy() const;

  [[nodiscard]] Contract GetContract() const { return contract_; }

  [[nodiscard]] std::vector<int> UidHistory() const {
    std::vector<int> uid_history;
    uid_history.reserve(move_history_.size());
    for (const auto& item : move_history_) {
      uid_history.push_back(parent_game_->GetMoveUid(item.move));
    }
    return uid_history;
  }

  // void SetDummyHand(const BridgeHand& dummy_hand);

  size_t operator()(const BridgeStateWithoutHiddenInfo& state) const {
    size_t hash_value = std::hash<int>()(state.contract_.Index());
    hash_value ^= std::hash<bool>()(state.is_dealer_vulnerable_) + 0x9e3779b9 +
                  (hash_value << 6) + (hash_value >> 2);
    hash_value ^= std::hash<bool>()(state.is_non_dealer_vulnerable_) +
                  0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
    for (const int uid : state.UidHistory()) {
      hash_value ^= std::hash<int>()(uid) + 0x9e3779b9 + (hash_value << 6) +
                    (hash_value >> 2);
    }
    // if (state.dummy_hand_.has_value()) {
    //   for (const auto &card : state.dummy_hand_.value().Cards()) {
    //     hash_value ^= std::hash<int>()(card.Index()) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
    //   }
    // }
    return hash_value;
  }

  std::string Serialize() const;

  static BridgeStateWithoutHiddenInfo Deserialize(
      const std::string& str, const std::shared_ptr<BridgeGame>& game);

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
  std::vector<int> scores_ = std::vector<int>(kNumPlayers);
  // std::optional<BridgeHand> dummy_hand_;

  void AdvanceToNextPlayer();
  [[nodiscard]] bool AuctionIsLegal(const BridgeMove& move) const;
  [[nodiscard]] bool PlayIsLegal(const BridgeMove& move) const;

  [[nodiscard]] std::string FormatVulnerability() const;
  [[nodiscard]] std::string FormatDeal() const;
  [[nodiscard]] std::string FormatAuction() const;
  [[nodiscard]] std::string FormatPlay() const;
  [[nodiscard]] std::string FormatResult() const;

  // BridgeHand OriginalDummyHand() const;

  void ScoreUp();
};

std::ostream& operator<<(std::ostream& stream,
                         const BridgeStateWithoutHiddenInfo& state);

}  // namespace bridge_learning_env

#endif /* PLAYCC_BRIDGE_STATE_WITHOUT_HIDDEN_INFO_H */
