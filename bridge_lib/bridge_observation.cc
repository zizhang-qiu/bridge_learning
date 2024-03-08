//
// Created by qzz on 2023/9/20.
//

#include "bridge_observation.h"

namespace bridge_learning_env {
int PlayerToOffset(const Player pid, const Player observer_pid) {
  const int direct_offset = pid - observer_pid;
  return direct_offset < 0 ? direct_offset + kNumPlayers : direct_offset;
}

void ChangeHistoryItemToObserverRelative(const Player observer_player,
                                         BridgeHistoryItem* item) {
  if (item->move.MoveType() == BridgeMove::kDeal) {
    REQUIRE(item->player == kChancePlayerId && item->deal_to_player > 0);
    item->deal_to_player =
        (item->deal_to_player - observer_player + kNumPlayers) % kNumPlayers;
  } else {
    REQUIRE(item->player >= 0);
    item->player = (item->player - observer_player + kNumPlayers) % kNumPlayers;
  }
}

BridgeObservation::BridgeObservation(const BridgeState& state,
                                     const Player observing_player)
    : cur_player_offset_(
          PlayerToOffset(state.CurrentPlayer(), observing_player)),
      observing_player_(observing_player),
      current_phase_(state.CurrentPhase()),
      legal_moves_(state.LegalMoves(observing_player)),
      contract_(state.GetContract()),
      tricks_(state.Tricks()),
      num_declarer_tricks_(state.NumDeclarerTricks()),
      parent_game_(state.ParentGame()) {
  hands_.reserve(kNumPlayers);
  hands_.push_back(state.Hands()[observing_player_]);
  for (int offset = 1; offset < kNumPlayers; ++offset) {
    hands_.push_back(state.Hands()[(observing_player_ + offset) % kNumPlayers]);
  }

  const auto& history = state.History();
  for (int i = kNumCards; i < history.size(); ++i) {
    auto item = history[i];

    ChangeHistoryItemToObserverRelative(observing_player_, &item);
    switch (item.move.MoveType()) {
      case BridgeMove::kAuction:
        auction_history_.push_back(item);
        break;
      case BridgeMove::kPlay:
        play_history_.push_back(item);
        break;
      case BridgeMove::kDeal:
        deal_history_.push_back(item);
        break;
      default:
        // Should be impossible.
        std::cerr << "History contains invalid move." << std::endl;
        std::abort();
    }
  }

  is_player_vulnerable_ = parent_game_->IsPlayerVulnerable(observing_player_);
  is_opponent_vulnerable_ =
      parent_game_->IsPlayerVulnerable((observing_player_ + 1) % kNumPlayers);
}

std::string BridgeObservation::ToString() const {
  std::string rv;
  rv += StrCat("Hand:\n", hands_[0].ToString());
  if (!auction_history_.empty()) {
    rv += "\nAuction history:\n";
    // for (const auto& item : auction_history_) {
    //   rv += StrCat(item.ToString(), "");
    // }
    rv += "\nWest  North East  South\n      ";
    for (int i = 0; i < auction_history_.size(); ++i) {
      if (i % kNumPlayers == kNumPlayers - 1) {
        rv.push_back('\n');
      }
      rv +=
          StrFormat("%-6s", auction_history_[i].move.AuctionToString().c_str());
    }
  }

  if (!play_history_.empty()) {
    rv += "\nContract is " + contract_.ToString() + "\n\nN  E  S  W  N  E  S";
    Player player = (1 + contract_.declarer) % kNumPlayers;
    Trick trick{kInvalidPlayer, kNoTrump, 0};
    for (int i = 0; i < play_history_.size(); ++i) {
      if (i % kNumPlayers == 0) {
        if (i > 0) {
          player = trick.Winner();
        }
        rv += "\n";
        rv += std::string(3 * player, ' ');
      } else {
        player = (1 + player) % kNumPlayers;
      }

      const auto item = play_history_[i];
      const auto card = CardIndex(item.suit, item.rank);
      // A new trick
      if (i % kNumPlayers == 0) {
        trick = Trick(player, contract_.denomination, card);
      } else {
        trick.Play(player, card);
      }
      rv += CardString(card);
      rv += " ";
    }
    rv += "\n\nDeclarer tricks: ";
    rv += std::to_string(num_declarer_tricks_);
  }

  return rv;
}
}  // namespace bridge_learning_env
