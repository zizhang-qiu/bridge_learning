#include "bridge_state.h"
#include "utils.h"

#include <array>
#include <cassert>
#include <iostream>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>

#include "third_party/dds/src/Memory.h"
#include "third_party/dds/src/SolverIF.h"
#include "third_party/dds/src/TransTableL.h"

// Our preferred version of the double_dummy_solver defines a DDS_EXTERNAL
// macro to add a prefix to the exported symbols to avoid name clashes.
// In order to compile with versions of the double_dummy_solver which do not
// do this, we define DDS_EXTERNAL as an identity if it isn't already defined.
#ifndef DDS_EXTERNAL
#define DDS_EXTERNAL(x) x
#endif

namespace bridge {
BridgeState::BridgeState(bool is_dealer_vulnerable,
                         bool is_non_dealer_vulnerable)
    : phase_(Phase::kDeal),
      cur_player_(kChancePlayerId),
      history_(),
      is_vulnerable_{is_dealer_vulnerable, is_non_dealer_vulnerable},
      num_passes_(0),
      num_declarer_tricks_(0),
      num_cards_played_(0),
      contract_{0},
      first_bidder_(),
      holder_(),
      tricks_(),
      returns_(std::vector<double>(4)) {
  possible_contracts_.fill(true);
}

std::vector<Action> BridgeState::History() const {
  std::vector<Action> ret;
  ret.reserve(history_.size());
  for (const auto player_action : history_) {
    ret.push_back(player_action.action);
  }
  return ret;
}

std::string BridgeState::ActionToString(Player player, Action action) const {
  if (action < kBiddingActionBase) {
    return CardString(action);
  }
  return CallString(action - kBiddingActionBase);
}

std::array<std::string, kNumSuits> BridgeState::FormatHand(
    Player player, bool mark_voids,
    const std::array<std::optional<Player>, kNumCards> &deal) const {
  std::array<std::string, kNumSuits> cards;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    cards[suit].push_back(kSuitChar[suit]);
    cards[suit].push_back(' ');
    bool is_void = true;
    for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
      if (player == deal[CardIndex(Suit(suit), rank)]) {
        cards[suit].push_back(kRankChar[rank]);
        is_void = false;
      }
    }
    if (is_void && mark_voids) cards[suit] += "none";
  }
  return cards;
}

std::array<std::optional<Player>, kNumCards> BridgeState::OriginalDeal() const {
  assert(history_.size() > kNumCards);
  std::array<std::optional<Player>, kNumCards> original_deal;
  for (int i = 0; i < kNumCards; ++i) {
    original_deal[history_[i].action] = (i % kNumPlayers);
  }
  return original_deal;
}

std::string BridgeState::FormatDeal() const {
  std::array<std::array<std::string, kNumSuits>, kNumPlayers> cards;
  if (IsTerminal()) {
    auto deal = OriginalDeal();
    for (auto player : kAllSeats) {
      cards[player] = FormatHand(player, false, deal);
    }
  } else {
    for (auto player : kAllSeats) {
      cards[player] = FormatHand(player, false, holder_);
    }
  }
  constexpr int kColumnWidth = 8;
  std::string padding(kColumnWidth, ' ');
  std::string rv;

  // Format north
  for (const Suit suit : kAllSuitsReverse) {
    rv += padding;
    rv += cards[kNorth][static_cast<int>(suit)];
    rv += "\n";
  }

  // Format west and east
  for (const Suit suit : kAllSuitsReverse) {
    rv += StrFormat("%-8s", cards[kWest][static_cast<int>(suit)].c_str());
    rv += padding;
    rv += cards[kEast][static_cast<int>(suit)];
    rv += "\n";
  }

  // Format south
  for (const Suit suit : kAllSuitsReverse) {
    rv += padding;
    rv += cards[kSouth][static_cast<int>(suit)];
    rv += "\n";
  }
  return rv;
}

std::string BridgeState::FormatVulnerability() const {
  std::string vul = is_vulnerable_[0] ? (is_vulnerable_[1] ? "All" : "N/S")
                                      : (is_vulnerable_[1] ? "E/W" : "None");
  return "Vul: " + vul + "\n";
}

std::string BridgeState::FormatAuction(bool trailing_query) const {
  assert(history_.size() > kNumCards);
  std::string rv = "\nWest  North East  South\n      ";
  for (int i = kNumCards; i < history_.size() - num_cards_played_; ++i) {
    // New line
    if (i % kNumPlayers == kNumPlayers - 1) {
      rv.push_back('\n');
    }
    rv +=
        StrFormat("%-6s", CallString(history_[i].action - kBiddingActionBase).c_str());
  }

  if (trailing_query) {
    if ((history_.size() - num_cards_played_) % kNumPlayers ==
        kNumPlayers - 1) {
      rv.push_back('\n');
    }
    rv.push_back('?');
  }
  return rv;
}

std::string BridgeState::FormatPlay() const {
  assert(num_cards_played_ > 0);
  std::string rv = "\n\nN  E  S  W  N  E  S";
  Trick trick{kInvalidPlayer, kNoTrump, 0};
  // opening lead by the player left to declarer
  Player player = (1 + contract_.declarer) % kNumPlayers;

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

    const int card = history_[history_.size() - num_cards_played_ + i].action;
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

std::string BridgeState::FormatResult() const {
  assert(IsTerminal());
  std::string rv;
  rv += "\nScore: N/S ";
  rv += std::to_string(returns_[kNorth]);
  rv += " E/W ";
  rv += std::to_string(returns_[kEast]);
  return rv;
}

std::string BridgeState::ToString() const {
  std::string rv = FormatVulnerability() + FormatDeal();
  if (history_.size() > kNumCards) {
    rv += FormatAuction(false);
  }
  if (num_cards_played_ > 0) {
    rv += FormatPlay();
  }
  if (IsTerminal()) {
    rv += FormatResult();
  }
  return rv;
}

std::vector<Action> BridgeState::LegalActions() const {
  switch (phase_) {
    case Phase::kDeal:return DealLegalActions();
    case Phase::kAuction:return BiddingLegalActions();
    case Phase::kPlay:return PlayLegalActions();
    default:return {};
  }
}

std::vector<Action> BridgeState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCards - history_.size());
  for (int i = 0; i < kNumCards; ++i) {
    if (!holder_[i].has_value()) {
      legal_actions.push_back(i);
    }
  }
  return legal_actions;
}

std::vector<Action> BridgeState::BiddingLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCalls);
  // Pass is always legal.
  legal_actions.push_back(kBiddingActionBase + kPass);

  int declarer_partnership = Partnership(contract_.declarer);
  int current_player_partnership = Partnership(cur_player_);

  if (contract_.level > 0 &&
      declarer_partnership != current_player_partnership &&
      contract_.double_status == kUndoubled) {
    legal_actions.push_back(kBiddingActionBase + kDouble);
  }

  if (contract_.level > 0 &&
      declarer_partnership == current_player_partnership &&
      contract_.double_status == kDoubled) {
    legal_actions.push_back(kBiddingActionBase + kRedouble);
  }

  for (Action bid = BidIndex(contract_.level, contract_.denomination) + 1;
       bid < kNumCalls; ++bid) {
    legal_actions.push_back(kBiddingActionBase + bid);
  }
  return legal_actions;
}

std::vector<Action> BridgeState::PlayLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCardsPerHand - num_cards_played_ / kNumPlayers);

  // Check if we can follow suit.
  if (num_cards_played_ % kNumPlayers != 0) {
    auto suit = CurrentTrick().LedSuit();
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      if (holder_[CardIndex(suit, rank)] == cur_player_) {
        legal_actions.push_back(CardIndex(suit, rank));
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Otherwise, we can play any of our cards.
  for (int card = 0; card < kNumCards; ++card) {
    if (holder_[card] == cur_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

void BridgeState::ApplyAction(Action action) {
  Player player = CurrentPlayer();
  DoApplyAction(action);
  history_.push_back({player, action});
}

void BridgeState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:return ApplyDealAction(action);
    case Phase::kAuction:return ApplyBiddingAction(action - kBiddingActionBase);
    case Phase::kPlay:return ApplyPlayAction(action);
    case Phase::kGameOver:std::cerr << "Cannot act in terminal states" << std::endl;
      std::exit(1);
  }
}

void BridgeState::ApplyDealAction(Action card) {
  holder_[card] = (history_.size() % kNumPlayers);
  if (history_.size() == kNumCards - 1) {
    phase_ = Phase::kAuction;
    cur_player_ = kNorth;
  }
}

void BridgeState::ApplyBiddingAction(Action call) {
  if (call == kPass) {
    ++num_passes_;
  } else {
    num_passes_ = 0;
  }

  int partnership = Partnership(cur_player_);

  if (call == kDouble) {
    assert(Partnership(contract_.declarer) != partnership);
    assert(contract_.double_status == kUndoubled);
    assert(contract_.level > 0);
    possible_contracts_[contract_.Index()] = false;
    contract_.double_status = kDoubled;
  } else if (call == kRedouble) {
    assert(Partnership(contract_.declarer == partnership));
    assert(contract_.double_status == kDoubled);
    possible_contracts_[contract_.Index()] = false;
    contract_.double_status = kRedoubled;
  } else if (call == kPass) {
    if (num_passes_ == kNumPlayers) {
      // Four consecutive passes can only happen if no-one makes a bid.
      // The hand is then over, and each side scores zero points.
      phase_ = Phase::kGameOver;
      possible_contracts_.fill(false);
      possible_contracts_[0] = true;
    } else if (num_passes_ == 3 && contract_.level > 0) {
      // After there has been a bid, three consecutive passes end the auction.
      possible_contracts_.fill(false);
      possible_contracts_[contract_.Index()] = true;
      phase_ = Phase::kPlay;
      // Opening lead.
      cur_player_ = (contract_.declarer + 1) % kNumPlayers;
      return;
    }
  } else {
    int bid_level = BidLevel(call);
    Denomination bid_denomination = BidDenomination(call);
    // A bid was made.
    assert(bid_level > contract_.level ||
        (bid_level == contract_.level &&
            bid_denomination > contract_.denomination));

    contract_.level = bid_level;
    contract_.denomination = bid_denomination;
    contract_.double_status = kUndoubled;
    if (!first_bidder_[partnership][contract_.denomination].has_value()) {
      // Partner cannot declare this denomination.
      first_bidder_[partnership][contract_.denomination] = cur_player_;
      const int partner = Partner(cur_player_);
      for (int level = contract_.level + 1; level <= kNumBidLevels; ++level) {
        for (DoubleStatus double_status : kAllDoubleStatus) {
          const Contract contract =
              Contract{level, contract_.denomination, double_status, partner};
          const int index = contract.Index();
          possible_contracts_[index] = false;
        }
      }
    }

    contract_.declarer =
        first_bidder_[partnership][contract_.denomination].value();
    // No lower contract is possible.
    std::fill(
        possible_contracts_.begin(),
        possible_contracts_.begin() +
            Contract{contract_.level, contract_.denomination, kUndoubled, 0}
                .Index(),
        false);

    // No-one else can declare this precise contract.
    for (int player = 0; player < kNumPlayers; ++player) {
      if (player != cur_player_) {
        for (DoubleStatus double_status : {kUndoubled, kDoubled, kRedoubled}) {
          possible_contracts_[Contract{contract_.level, contract_.denomination,
                                       double_status, player}
              .Index()] = false;
        }
      }
    }
  }
  cur_player_ = (cur_player_ + 1) % kNumPlayers;
}

void BridgeState::ApplyPlayAction(Action card) {
  assert(holder_[card] == cur_player_);
  holder_[card] = std::nullopt;

  if (num_cards_played_ % kNumPlayers == 0) {
    // A new round
    CurrentTrick() = Trick(cur_player_, contract_.denomination, card);
  } else {
    CurrentTrick().Play(cur_player_, card);
  }

  const Player winner = CurrentTrick().Winner();
  ++num_cards_played_;

  if (num_cards_played_ % kNumPlayers == 0) {
    // Winner leads next round.
    cur_player_ = winner;
    if (Partnership(winner) == Partnership(contract_.declarer)) {
      ++num_declarer_tricks_;
    }
  } else {
    cur_player_ = (cur_player_ + 1) % kNumPlayers;
  }

  if (num_cards_played_ == kNumCards) {
    phase_ = Phase::kGameOver;
  }
  ScoreUp();
}

void BridgeState::ScoreUp() {
  bool is_declarer_vulnerable = is_vulnerable_[Partnership(contract_.declarer)];
  int declarer_score =
      Score(contract_, num_declarer_tricks_, is_declarer_vulnerable);
  for (Player player = 0; player < kNumPlayers; ++player) {
    returns_[player] = Partnership(player) == Partnership(contract_.declarer)
                       ? declarer_score
                       : -declarer_score;
  }
}

Player BridgeState::CurrentPlayer() const {
  if (phase_ == Phase::kDeal) {
    return kChancePlayerId;
  } else if (phase_ == Phase::kGameOver) {
    return kTerminalPlayerId;
  } else if (phase_ == Phase::kPlay &&
      Partnership(cur_player_) == Partnership(contract_.declarer)) {
    // Declarer chooses cards for both players.
    return contract_.declarer;
  } else {
    return cur_player_;
  }
}

int BridgeState::ContractIndex() const {
  assert(phase_ == Phase::kPlay || phase_ == Phase::kGameOver);
  return contract_.Index();
}

std::mutex dds_mutex;

std::vector<int> BridgeState::ScoreForContracts(
    int player, const std::vector<int> &contracts) const {
  // Storage for the number of tricks.
  std::array<std::array<int, kNumPlayers>, kNumDenominations> dd_tricks{};

  if (double_dummy_results_.has_value()) {
    // If we have already computed double-dummy results, use them.
    for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
      for (int trumps = 0; trumps < kNumDenominations; ++trumps) {
        dd_tricks[trumps][declarer] =
            double_dummy_results_->resTable[DDSStrainToDenomination(trumps)][declarer];
      }
    }
  } else {
    {
      // This performs some sort of global initialization; unclear
      // exactly what.
      std::lock_guard<std::mutex> lock(dds_mutex);
      DDS_EXTERNAL(SetMaxThreads)(0);
    }

    // Working storage for DD calculation.
    auto thread_data = std::make_unique<ThreadData>();
    auto transposition_table = std::make_unique<TransTableL>();
    transposition_table->SetMemoryDefault(95);   // megabytes
    transposition_table->SetMemoryMaximum(160);  // megabytes
    transposition_table->MakeTT();
    thread_data->transTable = transposition_table.get();

    // Which trump suits do we need to handle?
    std::set<int> suits;
    for (auto index : contracts) {
      const auto &contract = kAllContracts[index];
      if (contract.level > 0) suits.emplace(contract.denomination);
    }
    // Build the deal
    ::deal dl{};
    std::array<std::optional<Player>, kNumCards> deal;
    if (phase_ == Phase::kPlay || phase_ == Phase::kGameOver) {
      deal = OriginalDeal();
    } else {
      deal = holder_;
    }
    for (int suit = 0; suit < kNumSuits; ++suit) {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        const int pl = deal[CardIndex(Suit(suit), rank)].value();
        dl.remainCards[pl][SuitToDDSSuit(Suit(suit))] += 1 << (2 + rank);
      }
    }
    for (int k = 0; k <= 2; k++) {
      dl.currentTrickRank[k] = 0;
      dl.currentTrickSuit[k] = 0;
    }

    // Analyze for each trump suit.
    for (int suit : suits) {
      dl.trump = suit;
      transposition_table->ResetMemory(TT_RESET_NEW_TRUMP);

      // Assemble the declarers we need to consider.
      std::set<int> declarers;
      for (auto index : contracts) {
        const auto &contract = kAllContracts[index];
        if (contract.level > 0 && contract.denomination == suit)
          declarers.emplace(contract.declarer);
      }

      // Analyze the deal for each declarer.
      std::optional<Player> first_declarer;
      std::optional<int> first_tricks;
      for (int declarer : declarers) {
        ::futureTricks fut{};
        dl.first = (declarer + 1) % kNumPlayers;
        if (!first_declarer.has_value()) {
          // First time we're calculating this trump suit.
          const int return_code = SolveBoardInternal(
              thread_data.get(), dl,
              /*target=*/-1,    // Find max number of tricks
              /*solutions=*/1,  // Just the tricks (no card-by-card result)
              /*mode=*/2,       // Unclear
              &fut              // Output
          );
          if (return_code != RETURN_NO_FAULT) {
            char error_message[80];
            DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
            std::cerr << "double dummy solver: " << error_message << std::endl;
            std::exit(1);
          }
          dd_tricks[DDSStrainToDenomination(suit)][declarer] = 13 - fut.score[0];
          first_declarer = declarer;
          first_tricks = 13 - fut.score[0];
        } else {
          // Reuse data from last time.
          const int hint = Partnership(declarer) == Partnership(*first_declarer)
                           ? *first_tricks
                           : 13 - *first_tricks;
          const int return_code =
              SolveSameBoard(thread_data.get(), dl, &fut, hint);
          if (return_code != RETURN_NO_FAULT) {
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
  for (int contract_index : contracts) {
    const Contract &contract = kAllContracts[contract_index];
    const int declarer_score =
        (contract.level == 0)
        ? 0
        : Score(contract, dd_tricks[contract.denomination][contract.declarer],
                is_vulnerable_[Partnership(contract.declarer)]);
    scores.push_back(Partnership(contract.declarer) == Partnership(player)
                     ? declarer_score
                     : -declarer_score);
  }
  return scores;
}

std::array<std::array<int, kNumPlayers>, kNumDenominations> BridgeState::DoubleDummyResults(bool dds_order) const {
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

void BridgeState::ComputeDoubleDummyTricks() const {
  if (!double_dummy_results_.has_value()) {
    std::lock_guard<std::mutex> lock(dds_mutex);
    double_dummy_results_ = ddTableResults{};
    ddTableDeal dd_table_deal{};
    for (int suit = 0; suit < kNumSuits; ++suit) {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        const int player = holder_[CardIndex(Suit(suit), rank)].value();
        dd_table_deal.cards[player][SuitToDDSSuit(Suit(suit))] += 1 << (2 + rank);
      }
    }
    DDS_EXTERNAL(SetMaxThreads)(0);
    const int return_code = DDS_EXTERNAL(CalcDDtable)(
        dd_table_deal, &double_dummy_results_.value());
    if (return_code != RETURN_NO_FAULT) {
      char error_message[80];
      DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
      std::cerr << "double_dummy_solver:" << error_message << std::endl;
      std::exit(1);
    }
  }
}

std::vector<std::pair<Action, double>> BridgeState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  int num_cards_remaining = kNumCards - static_cast<int>(history_.size());
  outcomes.reserve(num_cards_remaining);
  const double p = 1.0 / static_cast<double>(num_cards_remaining);
  for (int card = 0; card < kNumCards; ++card) {
    if (!holder_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void BridgeState::SetDoubleDummyResults(const std::vector<int>& double_dummy_tricks) {
  assert(double_dummy_tricks.size()==kNumDenominations * kNumPlayers);
  auto double_dummy_results = ddTableResults{};
  for (auto denomination : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
    for (auto player : {kNorth, kEast, kSouth, kWest}) {
      auto index = denomination * kNumPlayers + player;
      double_dummy_results.resTable[DenominationToDDSStrain(denomination)][player] =
          double_dummy_tricks[index];
    }
  }
  double_dummy_results_ = double_dummy_results;
}

}  // namespace bridge