//
// Created by qzz on 2023/9/23.
//
#include "bridge_state.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>
#include "bridge_deck.h"

#ifndef DDS_EXTERNAL
#define DDS_EXTERNAL(x) x
#endif
namespace bridge_learning_env {

BridgeState::BridgeState(std::shared_ptr<BridgeGame> parent_game) :
    parent_game_(std::move(parent_game)),
    deck_(),
    hands_(kNumPlayers),
    current_player_(kChancePlayerId),
    phase_(Phase::kDeal),
    auction_tracker_(),
    contract_(),
    played_cards_(),
    num_cards_played_(0),
    num_declarer_tricks_(0),
    last_round_winner_(-1),
    scores_(kNumPlayers, 0) {
  is_dealer_vulnerable_ = parent_game_->IsDealerVulnerable();
  is_non_dealer_vulnerable_ = parent_game_->IsNonDealerVulnerable();
  dealer_ = parent_game_->Dealer();
}

void BridgeState::AdvanceToNextPlayer() {
  if (!deck_.Empty()) {
    current_player_ = kChancePlayerId;
  }
  else if (phase_ == Phase::kAuction) {
    if (move_history_.size() == kNumCards) {
      current_player_ = dealer_;
    }
    else {
      current_player_ = (current_player_ + 1) % kNumPlayers;
    }
  }
  else if (phase_ == Phase::kPlay) {
    if (num_cards_played_ == 0) {
      current_player_ = (contract_.declarer + 1) % kNumPlayers;
    }
    else if (num_cards_played_ % kNumPlayers == 0) {
      // Winner leads next round.
      current_player_ = last_round_winner_;
    }
    else {
      current_player_ = (current_player_ + 1) % kNumPlayers;
    }
  }
  else {
    current_player_ = kInvalidPlayer;
  }
}

bool BridgeState::DealIsLegal(const BridgeMove &move) const {
  if (phase_ != Phase::kDeal) {
    return false;
  }
  if (!deck_.CardInDeck(move.CardSuit(), move.CardRank())) {
    return false;
  }
  return true;
}

bool BridgeState::AuctionIsLegal(const BridgeMove &move) const {
  if (phase_ != Phase::kAuction) {
    return false;
  }
  return auction_tracker_.AuctionIsLegal(move, CurrentPlayer());
}

bool BridgeState::PlayIsLegal(const BridgeMove &move) const {
  if (phase_ != Phase::kPlay) {
    return false;
  }
  if (!hands_[current_player_].IsCardInHand({move.CardSuit(), move.CardRank()})) {
    return false;
  }
  if (num_cards_played_ % kNumPlayers == 0) {
    // The player who leads to a trick may play any card in his hand
    return true;
  }
  const auto suit = CurrentTrick().LedSuit();

  // If the suit of the card is the same as lead suit, it is legal.
  if (move.CardSuit() == suit) {
    return true;
  }

  // If neither of the player's cards is the same as lead suit, the card is
  // legal.
  bool can_follow = false;
  for (const auto c : hands_[current_player_].Cards()) {
    if (c.CardSuit() == suit) {
      can_follow = true;
      break;
    }
  }
  if (!can_follow) {
    return true;
  }
  return false;
}

bool BridgeState::MoveIsLegal(const BridgeMove &move) const {
  switch (move.MoveType()) {
    case BridgeMove::kAuction:
      return AuctionIsLegal(move);
    case BridgeMove::kPlay:
      return PlayIsLegal(move);
    case BridgeMove::kDeal:
      return DealIsLegal(move);
    default:
      return false;
  }
}

void BridgeState::ApplyMove(const BridgeMove &move) {
  REQUIRE(MoveIsLegal(move));
  BridgeHistoryItem history(move);
  history.player = current_player_;
  switch (move.MoveType()) {
    case BridgeMove::kDeal:
      history.deal_to_player = PlayerToDeal();
      history.suit = move.CardSuit();
      history.rank = move.CardRank();
      hands_[history.deal_to_player].AddCard(deck_.DealCard(move.CardSuit(), move.CardRank()));
      if (deck_.Empty()) {
        phase_ = Phase::kAuction;
      }
      break;
    case BridgeMove::kAuction:
      history.level = move.BidLevel();
      history.denomination = move.BidDenomination();
      history.other_call = move.OtherCall();
      auction_tracker_.ApplyAuction(move, history.player);
      if (auction_tracker_.IsAuctionTerminated()) {
        contract_ = auction_tracker_.Contract();
        if (contract_.level == 0) {
          phase_ = Phase::kGameOver;
          ScoreUp();
        }
        else {
          phase_ = Phase::kPlay;
        }
      }
      break;
    case BridgeMove::kPlay:
      history.suit = move.CardSuit();
      history.rank = move.CardRank();
      hands_[current_player_].RemoveFromHand(move.CardSuit(), move.CardRank(), &played_cards_);
      if (num_cards_played_ % kNumPlayers == 0) {
        // A new round
        CurrentTrick() = Trick(current_player_, contract_.denomination, CardIndex(move.CardSuit(), move.CardRank()));
      }
      else {
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
    default:
      std::abort(); // Should not be possible.
  }
  move_history_.push_back(history);
  AdvanceToNextPlayer();
}

Player BridgeState::PlayerToDeal() const {
  // Always start from North
  if (deck_.Empty()) {
    return -1;
  }
  return (kNumCards - deck_.Size()) % kNumPlayers;
}

std::pair<std::vector<BridgeMove>, std::vector<double>> BridgeState::ChanceOutcomes() const {
  std::pair<std::vector<BridgeMove>, std::vector<double>> rv;
  const int max_outcome_uid = ParentGame()->MaxChanceOutcomes();
  for (int uid = 0; uid < max_outcome_uid; ++uid) {
    if (BridgeMove move = ParentGame()->GetChanceOutcome(uid); MoveIsLegal(move)) {
      rv.first.push_back(move);
      rv.second.push_back(ChanceOutcomeProb(move));
    }
  }
  return rv;
}
double BridgeState::ChanceOutcomeProb(const BridgeMove &move) const {
  return static_cast<double>(deck_.CardInDeck(move.CardSuit(), move.CardRank())) / static_cast<double>(deck_.Size());
}

void BridgeState::ApplyRandomChance() {
  const auto chance_outcomes = ChanceOutcomes();
  REQUIRE(!chance_outcomes.second.empty());
  ApplyMove(ParentGame()->PickRandomChance(chance_outcomes));
}

std::vector<BridgeMove> BridgeState::LegalMoves(const Player player) const {
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

std::vector<BridgeMove> BridgeState::LegalMoves() const { return LegalMoves(current_player_); }

std::vector<BridgeHand> BridgeState::OriginalDeal() const {
  std::vector<BridgeHand> rv(kNumPlayers);
  for (int i = 0; i < kNumCards; ++i) {
    const auto item = move_history_[i];
    const BridgeCard card = BridgeCard(item.suit, item.rank);
    rv[item.deal_to_player].AddCard(card);
  }
  return rv;
}
std::unique_ptr<BridgeState> BridgeState::Child(const BridgeMove &move) const {
  std::unique_ptr<BridgeState> child = Clone();
  child->ApplyMove(move);
  return child;
}

std::string BridgeState::ToString() const {
  std::string rv = FormatVulnerability() + FormatDeal();
  if (move_history_.size() > kNumCards) {
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

std::string BridgeState::FormatVulnerability() const {
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

std::string BridgeState::FormatDeal() const {
  std::string rv = StrCat("Dealer is ", ParentGame()->Dealer(), ".\n");
  std::vector<BridgeHand> hands;
  if (IsTerminal()) {
    hands = OriginalDeal();
  }
  else {
    hands = hands_;
  }
  for (const Player player : kAllSeats) {
    rv += kPlayerChar[player];
    rv += ": ";
    rv += hands[player].ToString() + "\n";
  }
  return rv;
}

std::string BridgeState::FormatAuction() const {
  REQUIRE(move_history_.size() > ParentGame()->MaxChanceOutcomes());
  std::string rv = "\nWest  North East  South\n      ";
  for (int i = kNumCards; i < move_history_.size() - num_cards_played_; ++i) {
    if (i % kNumPlayers == kNumPlayers - 1 - dealer_) {
      rv.push_back('\n');
    }
    rv += StrFormat("%-6s", move_history_[i].move.AuctionToString().c_str());
  }
  return rv;
}

std::string BridgeState::FormatPlay() const {
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
    }
    else {
      player = (1 + player) % kNumPlayers;
    }

    const auto item = move_history_[move_history_.size() - num_cards_played_ + i];
    const auto card = CardIndex(item.suit, item.rank);
    // A new trick
    if (i % kNumPlayers == 0) {
      trick = Trick(player, contract_.denomination, card);
    }
    else {
      trick.Play(player, card);
    }
    rv += CardString(card);
    rv += " ";
  }
  rv += "\n\nDeclarer tricks: ";
  rv += std::to_string(num_declarer_tricks_);
  return rv;
}

void BridgeState::ScoreUp() {
  REQUIRE(IsTerminal());
  if (contract_.level == 0) {
    for (const Player pl : kAllSeats) {
      scores_[pl] = 0;
    }
  }
  else {
    const int declarer_score = Score(contract_, num_declarer_tricks_, is_dealer_vulnerable_);
    for (const Player pl : kAllSeats) {
      if (Partnership(pl) == Partnership(contract_.declarer)) {
        scores_[pl] = declarer_score;
      }
      else {
        scores_[pl] = -declarer_score;
      }
    }
  }
}
std::string BridgeState::FormatResult() const {
  REQUIRE(IsTerminal());
  std::string rv;
  rv += "\nScore: N/S ";
  rv += std::to_string(scores_[kNorth]);
  rv += " E/W ";
  rv += std::to_string(scores_[kEast]);
  return rv;
}

std::mutex m_dds;
void BridgeState::ComputeDoubleDummyTricks() const {
  if (!double_dummy_results_.has_value()) {
    std::lock_guard<std::mutex> lock(m_dds);
    double_dummy_results_ = ddTableResults{};
    ddTableDeal dd_table_deal{};
    for (Player pl : kAllSeats) {
      for (const auto card : hands_[pl].Cards()) {
        dd_table_deal.cards[pl][SuitToDDSSuit(card.CardSuit())] += 1 << (2 + card.Rank());
      }
    }

    DDS_EXTERNAL(SetMaxThreads)(0);
    if (const int return_code = DDS_EXTERNAL(CalcDDtable)(dd_table_deal, &double_dummy_results_.value());
        return_code != RETURN_NO_FAULT) {
      char error_message[80];
      DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
      std::cerr << "double_dummy_solver:" << error_message << std::endl;
      std::exit(1);
    }
  }
}
std::array<std::array<int, kNumPlayers>, kNumDenominations> BridgeState::DoubleDummyResults(bool dds_order) const {
  REQUIRE(phase_ >= Phase::kAuction);
  if (!double_dummy_results_.has_value()) {
    ComputeDoubleDummyTricks();
  }
  std::array<std::array<int, kNumPlayers>, kNumDenominations> double_dummy_results{};
  std::memcpy(double_dummy_results.data(), double_dummy_results_.value().resTable, sizeof(double_dummy_results));
  if (!dds_order) {
    std::swap(double_dummy_results[0], double_dummy_results[3]);
    std::swap(double_dummy_results[1], double_dummy_results[2]);
  }
  return double_dummy_results;
}
void BridgeState::SetDoubleDummyResults(const std::vector<int> &double_dummy_tricks) const {
  REQUIRE(double_dummy_tricks.size() == kNumDenominations * kNumPlayers);

  auto double_dummy_results = ddTableResults{};
  for (const auto denomination : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
    for (const auto player : {kNorth, kEast, kSouth, kWest}) {
      const auto index = denomination * kNumPlayers + player;
      double_dummy_results.resTable[DenominationToDDSStrain(denomination)][player] = double_dummy_tricks[index];
    }
  }
  double_dummy_results_ = double_dummy_results;
}

void BridgeState::SetDoubleDummyResults(const array<int, kNumPlayers * kNumDenominations> &double_dummy_tricks) const {
  auto double_dummy_results = ddTableResults{};
  for (const auto denomination : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
    for (const auto player : {kNorth, kEast, kSouth, kWest}) {
      const auto index = denomination * kNumPlayers + player;
      double_dummy_results.resTable[DenominationToDDSStrain(denomination)][player] = double_dummy_tricks[index];
    }
  }
  double_dummy_results_ = double_dummy_results;
}
std::vector<int> BridgeState::ScoreForContracts(const Player player, const vector<int> &contracts) const {
  // Storage for the number of tricks.
  std::array<std::array<int, kNumPlayers>, kNumDenominations> dd_tricks{};

  if (double_dummy_results_.has_value()) {
    // If we have already computed double-dummy results, use them.
    for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
      for (int trumps = 0; trumps < kNumDenominations; ++trumps) {
        dd_tricks[trumps][declarer] = double_dummy_results_->resTable[DDSStrainToDenomination(trumps)][declarer];
      }
    }
  }
  else {
    {
      // This performs some sort of global initialization; unclear
      // exactly what.
      std::lock_guard<std::mutex> lock(m_dds);
      DDS_EXTERNAL(SetMaxThreads)(0);
    }

    // Working storage for DD calculation.
    auto thread_data = std::make_unique<ThreadData>();
    auto transposition_table = std::make_unique<TransTableL>();
    transposition_table->SetMemoryDefault(95); // megabytes
    transposition_table->SetMemoryMaximum(160); // megabytes
    transposition_table->MakeTT();
    thread_data->transTable = transposition_table.get();

    // Which trump suits do we need to handle?
    std::set<int> suits;
    for (const auto index : contracts) {
      const auto &contract = kAllContracts[index];
      if (contract.level > 0) suits.emplace(contract.denomination);
    }
    // Build the deal
    ::deal dl{};
    for (const Player pl : kAllSeats) {
      for (const auto card : hands_[pl].Cards()) {
        dl.remainCards[pl][SuitToDDSSuit(card.CardSuit())] += 1 << (2 + card.Rank());
      }
    }
    for (int k = 0; k <= 2; k++) {
      dl.currentTrickRank[k] = 0;
      dl.currentTrickSuit[k] = 0;
    }

    // Analyze for each trump suit.
    for (const int suit : suits) {
      dl.trump = suit;
      transposition_table->ResetMemory(TT_RESET_NEW_TRUMP);

      // Assemble the declarers we need to consider.
      std::set<int> declarers;
      for (const auto index : contracts) {
        const auto &contract = kAllContracts[index];
        if (contract.level > 0 && contract.denomination == suit) declarers.emplace(contract.declarer);
      }

      // Analyze the deal for each declarer.
      std::optional<Player> first_declarer;
      std::optional<int> first_tricks;
      for (int declarer : declarers) {
        ::futureTricks fut{};
        dl.first = (declarer + 1) % kNumPlayers;
        if (!first_declarer.has_value()) {
          // First time we're calculating this trump suit.
          if (const int return_code = SolveBoardInternal(thread_data.get(),
                                                         dl,
                                                         /*target=*/-1, // Find max number of tricks
                                                         /*solutions=*/1, // Just the tricks (no card-by-card result)
                                                         /*mode=*/2, // Unclear
                                                         &fut // Output
              );
              return_code != RETURN_NO_FAULT) {
            char error_message[80];
            DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
            std::cerr << "double dummy solver: " << error_message << std::endl;
            std::exit(1);
          }
          dd_tricks[DDSStrainToDenomination(suit)][declarer] = 13 - fut.score[0];
          first_declarer = declarer;
          first_tricks = 13 - fut.score[0];
        }
        else {
          // Reuse data from last time.
          const int hint = Partnership(declarer) == Partnership(*first_declarer) ? *first_tricks : 13 - *first_tricks;
          if (const int return_code = SolveSameBoard(thread_data.get(), dl, &fut, hint);
              return_code != RETURN_NO_FAULT) {
            char error_message[80];
            DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
            std::cerr << "double dummy solver: " << error_message << std::endl;
            std::exit(1);
          }
          dd_tricks[DDSStrainToDenomination(suit)][declarer] = 13 - fut.score[0];
        }
      }
    }
  }
  // Compute the scores.
  std::vector<int> scores;
  scores.reserve(contracts.size());
  for (const int contract_index : contracts) {
    const Contract &contract = kAllContracts[contract_index];
    const int declarer_score = (contract.level == 0)
        ? 0
        : Score(contract, dd_tricks[contract.denomination][contract.declarer], IsPlayerVulnerable(contract.declarer));
    scores.push_back(Partnership(contract.declarer) == Partnership(player) ? declarer_score : -declarer_score);
  }
  return scores;
}
bool BridgeState::IsPlayerVulnerable(Player player) const {
  return Partnership(player) == Partnership(dealer_) ? is_dealer_vulnerable_ : is_non_dealer_vulnerable_;
}
std::vector<int> BridgeState::UidHistory() const {
  std::vector<int> uid_history;
  uid_history.reserve(move_history_.size());
  for (const auto item : move_history_) {
    if (item.move.MoveType() == BridgeMove::kDeal) {
      uid_history.push_back(parent_game_->GetChanceOutComeUid(item.move));
    }
    else {
      uid_history.push_back(parent_game_->GetMoveUid(item.move));
    }
  }
  return uid_history;
}
Player BridgeState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kDeal:
      return kChancePlayerId;
    case Phase::kAuction:
      return current_player_;
    case Phase::kPlay:
      if (Partnership(current_player_) == Partnership(contract_.declarer)) {
        return contract_.declarer;
      }
      return current_player_;
    case Phase::kGameOver:
      return kTerminalPlayerId;
    default:
      return kInvalidPlayer;
  }
}
bool BridgeState::IsDummyActing() const {
  if (phase_ != Phase::kPlay) {
    return false;
  }
  if (current_player_ == Partner(contract_.declarer)) {
    return true;
  }
  return false;
}
Player BridgeState::GetDummy() const {
  if (phase_ != Phase::kPlay) {
    return kInvalidPlayer;
  }
  return Partner(contract_.declarer);
}

} // namespace bridge_learning_env
