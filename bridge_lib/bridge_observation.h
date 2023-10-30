//
// Created by qzz on 2023/9/20.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_OBSERVATION_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_OBSERVATION_H_
#include "bridge_game.h"
#include "bridge_hand.h"
#include "bridge_state_2.h"
namespace bridge_learning_env {
int PlayerToOffset(Player pid, Player observer_pid);

class BridgeObservation {

public:
  BridgeObservation(const BridgeState2 &state, Player observing_player);
  // offset of current player from observing player.
  int CurPlayerOffset() const { return cur_player_offset_; }

  std::string ToString() const;

  const std::vector<BridgeHistoryItem>& AuctionHistory() const{return auction_history_;}

  const std::vector<BridgeHand> &Hands() const { return hands_; }

  const std::shared_ptr<BridgeGame> ParentGame() const { return parent_game_; }

  const std::vector<BridgeMove> &LegalMoves() const { return legal_moves_; }

  bool IsPlayerVulnerable() const{return is_player_vulnerable_;}

  bool IsOpponentVulnerable() const{return is_opponent_vulnerable_;}

  Phase CurrentPhase() const{return current_phase_;}

private:
  int cur_player_offset_; // offset of current_player_ from observing_player
  Player observing_player_;
  Phase current_phase_;
  // hands_[0] contains observing player's hand.
  std::vector<BridgeHand> hands_;
  std::vector<BridgeMove> legal_moves_;
  std::vector<BridgeHistoryItem> auction_history_;
  bool is_player_vulnerable_;
  bool is_opponent_vulnerable_;
  std::shared_ptr<BridgeGame> parent_game_ = nullptr;
};

} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_OBSERVATION_H_
