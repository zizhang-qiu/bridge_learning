//
// Created by qzz on 2023/11/28.
//

#include "bridge_state_without_hidden_info.h"

namespace bridge_learning_env {
BridgeStateWithoutHiddenInfo::BridgeStateWithoutHiddenInfo(const std::shared_ptr<BridgeGame> &parent_game,
                                                           const BridgeHand &dummy_hand) :
    parent_game_(parent_game),
    current_player_(parent_game_->Dealer()),
    phase_(Phase::kAuction),
    auction_tracker_(),
    contract_(),
    num_cards_played_(0),
    num_declarer_tricks_(0),
    last_round_winner_(kInvalidPlayer),
    scores_(kNumPlayers, 0) {
  is_dealer_vulnerable_ = parent_game_->IsDealerVulnerable();
  is_non_dealer_vulnerable_ = parent_game_->IsNonDealerVulnerable();
  dealer_ = parent_game_->Dealer();
  REQUIRE(dummy_hand.IsFullHand());
  dummy_hand_ = dummy_hand;
}

BridgeStateWithoutHiddenInfo::BridgeStateWithoutHiddenInfo(const std::shared_ptr<BridgeGame> &parent_game,
                                                           const Contract &contract,
                                                           const BridgeHand &dummy_hand) :
    parent_game_(parent_game),
    current_player_((1 + contract.declarer) % kNumPlayers),
    phase_(Phase::kPlay),
    auction_tracker_(),
    contract_(contract),
    num_cards_played_(0),
    num_declarer_tricks_(0),
    last_round_winner_(kInvalidPlayer),
    scores_(kNumPlayers, 0) {
  is_dealer_vulnerable_ = parent_game_->IsDealerVulnerable();
  is_non_dealer_vulnerable_ = parent_game_->IsNonDealerVulnerable();
  dealer_ = parent_game_->Dealer();
  REQUIRE(dummy_hand.IsFullHand());
  dummy_hand_ = dummy_hand;
}

BridgeStateWithoutHiddenInfo::BridgeStateWithoutHiddenInfo(const BridgeState &state) :
    num_cards_played_(0), num_declarer_tricks_(0), scores_(kNumPlayers, 0), auction_tracker_() {
  REQUIRE(state.CurrentPhase() >= Phase::kPlay);
  parent_game_ = state.ParentGame();
//  contract_ = state.GetContract();
  current_player_ = parent_game_->Dealer();
//  std::cout << "current_player: " << current_player_ << std::endl;

  phase_ = Phase::kAuction;
  if (state.IsDummyCardShown()) {
    dummy_hand_ = state.OriginalDeal()[state.GetDummy()];
  }
  is_dealer_vulnerable_ = parent_game_->IsDealerVulnerable();
  is_non_dealer_vulnerable_ = parent_game_->IsNonDealerVulnerable();
  dealer_ = parent_game_->Dealer();
  last_round_winner_ = kInvalidPlayer;
  const auto &history = state.History();
  for (const auto &item : history) {
    if (item.move.MoveType() != BridgeMove::kDeal)
//      std::cout << item.ToString() << std::endl;
      ApplyMove(item.move);
//    }
  }
}

void BridgeStateWithoutHiddenInfo::AdvanceToNextPlayer() {
  if (phase_ == Phase::kAuction) {
    if (move_history_.empty()) {
      current_player_ = dealer_;
    } else {
      current_player_ = (current_player_ + 1) % kNumPlayers;
    }
  } else if (phase_ == Phase::kPlay) {
    if (num_cards_played_ == 0) {
      current_player_ = (contract_.declarer + 1) % kNumPlayers;
    } else if (num_cards_played_ % kNumPlayers == 0) {
      // Winner leads next round.
      current_player_ = last_round_winner_;
    } else {
      current_player_ = (current_player_ + 1) % kNumPlayers;
    }
  } else {
    current_player_ = kInvalidPlayer;
  }
}
bool BridgeStateWithoutHiddenInfo::AuctionIsLegal(const BridgeMove &move) const {
  if (phase_ != Phase::kAuction) {
    return false;
  }
  return auction_tracker_.AuctionIsLegal(move, CurrentPlayer());
}

Player BridgeStateWithoutHiddenInfo::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kAuction:return current_player_;
    case Phase::kPlay:
      if (Partnership(current_player_) == Partnership(contract_.declarer)) {
        return contract_.declarer;
      }
      return current_player_;
    case Phase::kGameOver:return kTerminalPlayerId;
    default:return kInvalidPlayer;
  }
}

bool BridgeStateWithoutHiddenInfo::PlayIsLegal(const BridgeMove &move) const {
//  std::cout << "current_player: " << current_player_ << std::endl;
  if (phase_ != Phase::kPlay) {
    return false;
  }

  // A card is illegal if it has been played.
  for (const auto &history_item : move_history_) {
    if (history_item.move == move) {
      return false;
    }
  }

//  std::cout << "current_player: " << current_player_ << std::endl;
//  std::cout << "dummy: " << GetDummy() << std::endl;
//  std::cout << "has value: " << dummy_hand_.has_value() << std::endl;
//  if(dummy_hand_.has_value()){
//    std::cout << "dummy hand: " << dummy_hand_->ToString() << std::endl;
//  }
  // If dummy hand is known, check the card is in.
  if (dummy_hand_.has_value() && current_player_ == GetDummy()) {
    if (!dummy_hand_->IsCardInHand({move.CardSuit(), move.CardRank()})) return false;
  }

  // If a suit is void, the move is not legal.
  const auto &void_suits = VoidSuitsForPlayer(current_player_);
  if (void_suits[move.CardSuit()]) {
    return false;
  }

  return true;
}

bool BridgeStateWithoutHiddenInfo::MoveIsLegal(const BridgeMove &move) const {
  switch (move.MoveType()) {
    case BridgeMove::kAuction:return AuctionIsLegal(move);
    case BridgeMove::kPlay:return PlayIsLegal(move);
    default:return false;
  }
}
void BridgeStateWithoutHiddenInfo::ApplyMove(const BridgeMove &move) {
  if (!MoveIsLegal(move)) {
    std::cout << "state:\n" << ToString() << "encounter illegal move: " << move << std::endl;
  }
  REQUIRE(MoveIsLegal(move));
  BridgeHistoryItem history(move);
  history.player = current_player_;
  switch (move.MoveType()) {
    case BridgeMove::kAuction:history.level = move.BidLevel();
      history.denomination = move.BidDenomination();
      history.other_call = move.OtherCall();
      auction_tracker_.ApplyAuction(move, history.player);
      if (auction_tracker_.IsAuctionTerminated()) {
        contract_ = auction_tracker_.Contract();
        if (contract_.level == 0) {
          phase_ = Phase::kGameOver;
          ScoreUp();
        } else {
          phase_ = Phase::kPlay;
        }
      }
      break;
    case BridgeMove::kPlay:history.suit = move.CardSuit();
      history.rank = move.CardRank();
      if (dummy_hand_.has_value() && current_player_ == GetDummy()) {
        dummy_hand_->RemoveFromHand(move.CardSuit(), move.CardRank(), nullptr);
      }
      if (num_cards_played_ % kNumPlayers == 0) {
        // A new round
        CurrentTrick() = Trick(current_player_, contract_.denomination, CardIndex(move.CardSuit(), move.CardRank()));
      } else {
        CurrentTrick().Play(current_player_, CardIndex(move.CardSuit(), move.CardRank()));
      }
      {
        const Player winner = CurrentTrick().Winner();
        ++num_cards_played_;
        if (num_cards_played_ % kNumPlayers == 0) {
          last_round_winner_ = winner;
          if (Partnership(last_round_winner_) == Partnership(contract_.declarer)) {
            ++num_declarer_tricks_;
          }
        }
      }
      if (num_cards_played_ == kNumCards) {
        phase_ = Phase::kGameOver;
        ScoreUp();
      }
      break;
    default:std::abort(); // Should not be possible.
  }
  move_history_.push_back(history);
  AdvanceToNextPlayer();
}
void BridgeStateWithoutHiddenInfo::ScoreUp() {
  REQUIRE(IsTerminal());
  if (contract_.level == 0) {
    for (const Player pl : kAllSeats) {
      scores_[pl] = 0;
    }
  } else {
    const int declarer_score = Score(contract_, num_declarer_tricks_, is_dealer_vulnerable_);
    for (const Player pl : kAllSeats) {
      if (Partnership(pl) == Partnership(contract_.declarer)) {
        scores_[pl] = declarer_score;
      } else {
        scores_[pl] = -declarer_score;
      }
    }
  }
}

BridgeStateWithoutHiddenInfo BridgeStateWithoutHiddenInfo::Child(const BridgeMove &move) const {
  BridgeStateWithoutHiddenInfo child(*this);
  child.ApplyMove(move);
  return child;
}
std::vector<BridgeMove> BridgeStateWithoutHiddenInfo::LegalMoves(Player player) const {
  std::vector<BridgeMove> legal_moves;
  // kChancePlayer=-1 must be handled by ChanceOutcome.
  REQUIRE(player >= 0 && player < kNumPlayers);
  if (player != current_player_) {
    return legal_moves;
  }

  const int max_move_uid = ParentGame()->MaxMoves();
  for (int uid = 0; uid < max_move_uid; ++uid) {
    if (BridgeMove move = ParentGame()->GetMove(uid); MoveIsLegal(move)) {
      legal_moves.push_back(move);
    }
  }
  return legal_moves;
}

std::string BridgeStateWithoutHiddenInfo::FormatDeal() const {
  std::string rv = StrCat("Dealer is ", ParentGame()->Dealer(), ".\n");

  // Show dummy's cards if opening lead has been made.
  for (const Player player : kAllSeats) {
    rv += kPlayerChar[player];
    rv += ": ";
    if (player == GetDummy() && num_cards_played_ > 0 && dummy_hand_.has_value()) {
      rv += dummy_hand_->ToString();
    }
    rv += "\n";
  }
  return rv;
}

std::string BridgeStateWithoutHiddenInfo::FormatAuction() const {
  REQUIRE(!move_history_.empty());
  if (move_history_[0].move.MoveType() != BridgeMove::kAuction) {
    return {};
  }
  std::string rv = "\nWest  North East  South\n      ";
  // Start from 0 since there is no dealing phase.
  for (int i = 0; i < move_history_.size() - num_cards_played_; ++i) {
    if (i % kNumPlayers == kNumPlayers - 1 - dealer_) {
      rv.push_back('\n');
    }
    rv += StrFormat("%-6s", move_history_[i].move.AuctionToString().c_str());
  }
  return rv;
}
std::string BridgeStateWithoutHiddenInfo::FormatPlay() const {
  REQUIRE(num_cards_played_ > 0);
  std::string rv = "\nContract is " + contract_.ToString() + "\n\nN  E  S  W  N  E  S";
  Player player = (1 + contract_.declarer) % kNumPlayers;
  Trick trick{kInvalidPlayer, kNoTrump, 0};
  for (int i = 0; i < num_cards_played_; ++i) {
    if (i % kNumPlayers == 0) {
      if (i > 0) {
        player = trick.Winner();
      }
      rv += "\n";
      rv += std::string(3 * player, ' ');
    } else {
      player = (1 + player) % kNumPlayers;
    }

    const auto item = move_history_[move_history_.size() - num_cards_played_ + i];
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
  return rv;
}
std::string BridgeStateWithoutHiddenInfo::FormatResult() const {
  REQUIRE(IsTerminal());
  std::string rv;
  rv += "\nScore: N/S ";
  rv += std::to_string(scores_[kNorth]);
  rv += " E/W ";
  rv += std::to_string(scores_[kEast]);
  return rv;
}
std::string BridgeStateWithoutHiddenInfo::FormatVulnerability() const {
  const bool is_ns_vulnerable =
      Partnership(dealer_) == Partnership(kNorth) ? is_dealer_vulnerable_ : is_non_dealer_vulnerable_;
  const bool is_ew_vulnerable =
      Partnership(dealer_) == Partnership(kEast) ? is_dealer_vulnerable_ : is_non_dealer_vulnerable_;
  if (is_ns_vulnerable && is_ew_vulnerable) {
    return "Vul: All\n";
  }
  if (is_ns_vulnerable) {
    return "Vul: N/S\n";
  }
  if (is_ew_vulnerable) {
    return "Vul: E/W\n";
  }
  return "Vul: None\n";
}

std::string BridgeStateWithoutHiddenInfo::ToString() const {
  std::string rv = FormatVulnerability() + FormatDeal();
  if (!move_history_.empty()) {
    rv += FormatAuction();
  }
  if (num_cards_played_ > 0) {
    rv += FormatPlay();
  }
  if (IsTerminal()) {
    rv += FormatResult();
  }
  return rv;
}
BridgeStateWithoutHiddenInfo BridgeStateWithoutHiddenInfo::FromBridgeState(const BridgeState &state,
                                                                           const bool with_bidding_history) {
  switch (state.CurrentPhase()) {
    case Phase::kDeal:return BridgeStateWithoutHiddenInfo(state.ParentGame(), {});
    case Phase::kPlay: {
      const auto &dummy_hand = state.Hands()[state.GetDummy()];
      if (!with_bidding_history) {
        const Contract contract = state.GetContract();
        const auto &history = state.History();
        BridgeStateWithoutHiddenInfo res_state{state.ParentGame(), contract, dummy_hand};
        for (const auto &item : history) {
          if (item.move.MoveType() == BridgeMove::kPlay) {
            res_state.ApplyMove(item.move);
          }
        }
        return res_state;
      }
    }
    case Phase::kAuction:
    case Phase::kGameOver:std::cout << "dummy: " << state.GetDummy() << std::endl;
      const auto &dummy_hand = state.Hands()[state.GetDummy()];
      std::cout << "dummy hand: " << dummy_hand.ToString() << std::endl;
      BridgeStateWithoutHiddenInfo res_state{state.ParentGame(), dummy_hand};
      const auto &history = state.History();
      for (size_t i = kNumCards; i < history.size(); ++i) {
        res_state.ApplyMove(history[i].move);
      }
      return res_state;
  }
  std::abort();
}
std::array<bool, kNumSuits> BridgeStateWithoutHiddenInfo::VoidSuitsForPlayer(const Player player) const {
  REQUIRE(player >= 0 && player < kNumPlayers);
  std::array<bool, kNumSuits> void_suit{false, false, false, false};
  const auto &play_history = PlayHistory();

  Suit led_suit = Suit::kInvalidSuit;
  for (int i = 0; i < num_cards_played_; ++i) {
    const int num_this_trick = i / kNumPlayers;
    const auto item = play_history[i];
    const Player this_move_player = item.player;

    const bool is_this_move_lead = (i % kNumPlayers == 0);
    if (is_this_move_lead) {
      led_suit = tricks_[num_this_trick].LedSuit();
    }

    if (this_move_player != player) {
      continue;
    }

    // Check if the player follows the led suit.
    if (item.move.CardSuit() != led_suit) {
      void_suit[led_suit] = true;
    }
  }
  return void_suit;
}
std::vector<BridgeHistoryItem> BridgeStateWithoutHiddenInfo::PlayHistory() const {
  std::vector<BridgeHistoryItem> play_history;
  for (const auto &item : move_history_) {
    if (item.move.MoveType() == BridgeMove::kPlay) {
      play_history.push_back(item);
    }
  }
  return play_history;
}
Player BridgeStateWithoutHiddenInfo::GetDummy() const {
  if (phase_ < Phase::kPlay) {
    return kInvalidPlayer;
  }
  return Partner(contract_.declarer);
}
void BridgeStateWithoutHiddenInfo::SetDummyHand(const BridgeHand &dummy_hand) {
  REQUIRE(dummy_hand.IsFullHand());
  REQUIRE(num_cards_played_ <= 1);
  dummy_hand_ = dummy_hand;
}
std::string BridgeStateWithoutHiddenInfo::Serialize() const {
  std::string rv{};
  for (int uid : UidHistory()) {
    rv += std::to_string(uid) + "\n";
  }
  if (dummy_hand_.has_value()) {
    std::string dummy = "Dummy Hand\n";
    auto original_dummy_hand = OriginalDummyHand();
    auto dummy_cards = original_dummy_hand.Cards();
    for (const auto &card : dummy_cards) {
      dummy += std::to_string(card.Index()) + "\n";
    }
    rv += dummy;
  }
  return rv;
}
BridgeStateWithoutHiddenInfo BridgeStateWithoutHiddenInfo::Deserialize(const std::string &str,
                                                                       const std::shared_ptr<BridgeGame> &game) {
  BridgeStateWithoutHiddenInfo state{};
  state.parent_game_ = game;
  std::vector<std::string> lines = StrSplit(str, '\n');

  const auto separator = std::find(lines.begin(), lines.end(), "Dummy Hand");
  if (separator != lines.end()) {
    auto it = separator;
    BridgeHand dummy_hand{};
    while (++it != lines.end()) {
      if (it->empty()) continue;
      const int card_index = std::stoi(*it);
      const BridgeCard card{CardSuit(card_index), CardRank(card_index)};
      dummy_hand.AddCard(card);
    }
    state.dummy_hand_ = dummy_hand;
    if (dummy_hand.Cards().size() != kNumCardsPerHand) {
      std::cout << "dummy hand error, lines=\n";
      std::cout << str << std::endl;
      std::cout << "dummy hand:\n";
      std::cout << dummy_hand.Cards() << std::endl;
    }
  }
  state.is_dealer_vulnerable_ = state.parent_game_->IsDealerVulnerable();
  state.is_non_dealer_vulnerable_ = state.parent_game_->IsNonDealerVulnerable();
  state.dealer_ = state.parent_game_->Dealer();
  state.last_round_winner_ = kInvalidPlayer;

  for (int i = 0; i < std::distance(lines.begin(), separator); ++i) {
//    std::cout << i << ", " << lines[i] << std::endl;
    if (lines[i].empty()) continue;
    int action_uid = std::stoi(lines[i]);
    BridgeMove move{};
    move = game->GetMove(action_uid);
    state.ApplyMove(move);
  }
  return state;
}
BridgeHand BridgeStateWithoutHiddenInfo::OriginalDummyHand() const {
  if (!dummy_hand_.has_value()) {
    return {};
  }
  auto hand = dummy_hand_.value();
  for (const auto &item : move_history_) {
    if (item.player == GetDummy() && item.move.MoveType() == BridgeMove::Type::kPlay) {
      const auto card = BridgeCard{item.move.CardSuit(), item.move.CardRank()};
      hand.AddCard(card);
    }
  }
  return hand;
}

bool operator==(const BridgeHand &lhs, const BridgeHand &rhs) {
  return lhs.Cards() == rhs.Cards();
}
std::ostream &operator<<(ostream &stream, const BridgeStateWithoutHiddenInfo &state) {
  return stream << state.ToString();
}
} // namespace bridge_learning_env
