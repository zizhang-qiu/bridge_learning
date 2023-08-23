#ifndef BRIDGE_STATE
#define BRIDGE_STATE
#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "bridge_game.h"
#include "bridge_scoring.h"
#include "bridge_utils.h"
#include "trick.h"

namespace bridge {
class BridgeState {
 public:
  BridgeState(bool is_dealer_vulnerable, bool is_non_dealer_vulnerable);

  bool IsTerminal() const { return phase_ == Phase::kGameOver; }

  int CurrentPhase() const { return static_cast<int>(phase_); }

  int ContractIndex() const;

  Player CurrentPlayer() const;

  std::string ActionToString(Player player, Action action) const;

  void ApplyAction(Action action);

  std::vector<Action> LegalActions() const;

  std::string ToString() const;

 private:
  // Format a player's hand
  std::array<std::string, kNumSuits> FormatHand(
      Player player, bool mark_voids,
      const std::array<std::optional<Player>, kNumCards>& deal) const;

  // Get original deal if any cards are played
  std::array<std::optional<Player>, kNumCards> OriginalDeal() const;
  std::string FormatDeal() const;
  std::string FormatVulnerability() const;
  std::string FormatAuction(bool trailing_query) const;
  std::string FormatPlay() const;
  std::string FormatResult() const;

  Trick& CurrentTrick() { return tricks_[num_cards_played_ / kNumPlayers]; }
  const Trick& CurrentTrick() const {
    return tricks_[num_cards_played_ / kNumPlayers];
  }

  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BiddingLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  
  void DoApplyAction(Action action);
  void ApplyDealAction(Action card);
  void ApplyBiddingAction(Action call);
  void ApplyPlayAction(Action card);

  void ScoreUp();

  enum class Phase { kDeal, kAuction, kPlay, kGameOver };
  Phase phase_;
  Player cur_player_;
  std::vector<PlayerAction> history_;
  bool is_vulnerable_[kNumPartnerships];

  // Tracks number of consecutive passes.
  int num_passes_;
  int num_declarer_tricks_;
  int num_cards_played_;
  Contract contract_;

  // Tracks for each denomination and partnership, who bid first, in order to
  // determine the declarer.
  std::array<std::array<std::optional<Player>, kNumDenominations>,
             kNumPartnerships>
      first_bidder_;

  // Tracks holder for each card.
  std::array<std::optional<Player>, kNumCards> holder_;

  std::array<bool, kNumContracts> possible_contracts_;

  std::array<Trick, kNumTricks> tricks_;

  std::vector<double> returns_;
};
}  // namespace bridge

#endif /* BRIDGE_STATE */
