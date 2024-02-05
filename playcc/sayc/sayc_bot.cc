//
// Created by qzz on 24-1-31.
//

#include "sayc_bot.h"

#include "utils.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "playcc/utils.h"

namespace sayc {

bool IsMajorSuit(const ble::Suit suit) {
  return suit == ble::Suit::kHeartsSuit || suit == ble::Suit::kSpadesSuit;
}

bool IsMinorSuit(const ble::Suit suit) {
  return suit == ble::Suit::kClubsSuit || suit == ble::Suit::kDiamondsSuit;
}

ble::BridgeMove SAYCBot::Step(const ble::BridgeObservation& obs) {
  InformObservation(obs);
  std::vector<ble::BridgeMove> possible_moves;
  const FitStatus can_open = LoadConstraint("can_open", {})->Fits(
      hand_analyzer_, obs, belief_);
  // std::cout << "can open" << std::endl;
  if (can_open.IsCertain()) {
    // Check NoTrump opening.
    const ble::BridgeMove no_trump_bid = NoTrumpOpening(obs);
    if (no_trump_bid.IsValid()) {
      // possible_moves.push_back(no_trump_bid);
      return no_trump_bid;
    }

    // Check one level opening
    const ble::BridgeMove one_level_bid = OneLevelOpening(obs);
    if (one_level_bid.IsValid()) {
      possible_moves.push_back(one_level_bid);
    }

  }
  if (!possible_moves.empty()) {
    return UniformSample(possible_moves, rng_);
  }
  return ConstructAuctionMoveFromString("Pass");
}

void SAYCBot::InformObservation(const ble::BridgeObservation& obs) {
  if (!hand_analyzer_.Hand().IsFullHand()) {
    hand_analyzer_.SetHand(obs.Hands()[0]);
  }
  SPIEL_CHECK_EQ(internal_history_,
                 std::vector<ble::BridgeHistoryItem>(obs.AuctionHistory().begin(
                 ), obs.AuctionHistory().begin() + internal_history_.size()));
  // [TODO] update belief.

}

void SAYCBot::Restart() {
  hand_analyzer_ = HandAnalyzer{};
  internal_history_.clear();
  belief_.fill({});
}

int SAYCBot::GetSeat(const ble::BridgeObservation& obs) const {
  const ble::Player dealer = obs.ParentGame()->Dealer();
  const ble::Player observing_player = obs.ObservingPlayer();
  return observing_player >= dealer
           ? observing_player - dealer
           : (observing_player + ble::kNumPlayers) - dealer;
}

ble::BridgeMove
SAYCBot::NoTrumpOpening(const ble::BridgeObservation& obs) const {
  if (const FitStatus can_open_1nt = LoadConstraint("1NT", {})->Fits(
      hand_analyzer_, obs, belief_); can_open_1nt.IsCertain()) {
    return ConstructAuctionMoveFromString("1NT");
  }

  if (const FitStatus can_open_2nt = LoadConstraint("2NT", {})->Fits(
      hand_analyzer_, obs, belief_); can_open_2nt.IsCertain()) {
    return ConstructAuctionMoveFromString("2NT");
  }

  if (const FitStatus can_open_3nt = LoadConstraint("3NT", {})->Fits(
      hand_analyzer_, obs, belief_); can_open_3nt.IsCertain()) {
    return ConstructAuctionMoveFromString("3NT");
  }

  // If no NoTrump constraint are satisfied, return an invalid move.
  return {};
}

ble::BridgeMove SAYCBot::OneLevelOpening(
    const ble::BridgeObservation& obs) const {
  const int seat = GetSeat(obs);
  // First and second seat, rule of 20.
  if (seat == 0 || seat == 1) {
    const auto rule_of_20 = LoadConstraint("rule_of_20", {});
    const FitStatus rule_of_20_status = rule_of_20->Fits(
        hand_analyzer_, obs, belief_);
    const auto one_level_open_hcp_constraint = LoadConstraint(
        "one_level_open_hcp", {});
    const FitStatus hcp_status = one_level_open_hcp_constraint->Fits(
        hand_analyzer_, obs, belief_);
    if (rule_of_20_status.IsCertain() && hcp_status.IsCertain()) {
      const auto move = OneLevelOpeningImpl(obs);
      return move;
    }
    return {};

  }

  // [TODO] Third seat.
  if (seat == 2) {}

  // Fourth seat.
  if (seat == 3) {

    const auto rule_of_15 = LoadConstraint("rule_of_15", {});
    const FitStatus status = rule_of_15->Fits(hand_analyzer_, obs, belief_);

    if (status.IsCertain()) {
      const auto move = OneLevelOpeningImpl(obs);
      return move;
    }
    return {};

  }

  return {};
}

ble::BridgeMove SAYCBot::OneLevelOpeningImpl(
    const ble::BridgeObservation& obs) const {
  const auto suit_length = hand_analyzer_.GetSuitLength();
  // Special case, 4-4 in minor, open 1D.
  if (suit_length[ble::Suit::kDiamondsSuit] == 4 &&
      suit_length[ble::Suit::kClubsSuit] == 4) {
    return ConstructAuctionMoveFromString("1D");
  }

  // Spiecial case, 4-4-3-2, open 1D
  if (suit_length == kOneDiamondSpecialCase) {
    return ConstructAuctionMoveFromString("1D");
  }

  // Special case, Holding 3-3 in the minors and no five-card major, open 1C.
  if (suit_length[ble::Suit::kDiamondsSuit] == 3 &&
      suit_length[ble::Suit::kClubsSuit] == 3 &&
      suit_length[ble::Suit::kHeartsSuit] != 5 &&
      suit_length[ble::Suit::kDiamondsSuit] != 5) {
    return ConstructAuctionMoveFromString("1C");
  }

  // Get longest suit length
  const auto sorted_suit_length_with_suit = hand_analyzer_.
      GetSortedSuitLengthWithSuits();

  const auto [longest_length, longest_suits] = sorted_suit_length_with_suit[0];

  if (longest_length >= kMinLengthForMajorOpening) {
    // If we have five card suits, bid highest suit.
    return ConstructBidMove(longest_suits[0], 1);
  } else {
    // Otherwise, bid highest minor suit.
    const auto iter = std::find_if(longest_suits.begin(), longest_suits.end(),
                                   [](const ble::Suit suit) {
                                     return IsMinorSuit(suit);
                                   });
    ble::Suit suit;
    if (iter == longest_suits.end()) {
      // We may have 4-4 in major suits. So bid second longest suit.
      suit = sorted_suit_length_with_suit[1].second[0];
    } else {
      suit = *iter;
    }
    return ConstructBidMove(suit, 1);
  }
}

}