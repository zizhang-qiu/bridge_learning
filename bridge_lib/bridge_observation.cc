//
// Created by qzz on 2023/9/20.
//

#include "bridge_observation.h"

namespace bridge_learning_env {

int PlayerToOffset(Player pid, Player observer_pid) {
  int direct_offset = pid - observer_pid;
  return direct_offset < 0 ? direct_offset + kNumPlayers : direct_offset;
}

void ChangeHistoryItemToObserverRelative(Player observer_player,
                                         BridgeHistoryItem *item) {
  if (item->move.MoveType() == BridgeMove::kDeal) {
    REQUIRE(item->player == kChancePlayerId && item->deal_to_player > 0);
    item->deal_to_player =
        (item->deal_to_player - observer_player + kNumPlayers) % kNumPlayers;
  } else {
    REQUIRE(item->player >= 0);
    item->player = (item->player - observer_player + kNumPlayers) % kNumPlayers;
  }
}

BridgeObservation::BridgeObservation(const BridgeState &state,
                                     Player observing_player)
    : observing_player_(observing_player),
      cur_player_offset_(
          PlayerToOffset(state.CurrentPlayer(), observing_player)),
      legal_moves_(state.LegalMoves(observing_player)),
      parent_game_(state.ParentGame()),
      current_phase_(state.CurrentPhase()){
  hands_.reserve(kNumPlayers);
  hands_.push_back(state.Hands()[observing_player_]);
  for (int offset = 1; offset < kNumPlayers; ++offset) {
    hands_.push_back(state.Hands()[(observing_player_ + offset) % kNumPlayers]);
  }

  const auto &history = state.History();
  for (int i = kNumCards; i < history.size(); ++i) {
    auto item = history[i];
    if (item.move.MoveType() != BridgeMove::kAuction) {
      break;
    }
    ChangeHistoryItemToObserverRelative(observing_player_, &item);
    auction_history_.push_back(item);
  }

  is_player_vulnerable_ = parent_game_->IsPlayerVulnerable(observing_player_);
  is_opponent_vulnerable_ =
      parent_game_->IsPlayerVulnerable((observing_player_ + 1) % kNumPlayers);
}

std::string BridgeObservation::ToString() const {
  std::string rv;
  rv += StrCat("Hand:\n", hands_[0].ToString());
  rv += "\nAuction history:\n";
  for (const auto &item : auction_history_) {
    rv += StrCat(item.ToString(), "");
  }
  return rv;
}
} // namespace bridge