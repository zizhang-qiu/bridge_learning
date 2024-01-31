//
// Created by qzz on 24-2-1.
//
#include "playcc/utils.h"
#include "playcc/sayc/constraints.h"
#include "playcc/sayc/utils.h"

namespace sayc {
void BalancedHandConstraintTest() {
  std::mt19937 rng;
  const auto constraint = LoadConstraint("balanced_hand", {});
  const std::vector<std::string> card_strings = {"SK", "S2", "HK",
                                                 "H9", "H8", "H7",
                                                 "H3", "DA", "DQ",
                                                 "D8", "CK", "CQ",
                                                 "C4"
  };
  const auto state = ConstructStateFromCardStrings({
                                                       {
                                                           card_strings,
                                                           {}, {}, {}
                                                       }}, ble::default_game,
                                                   rng);
  const ble::BridgeObservation obs{state};
  const auto hand = obs.Hands()[0];
  const HandAnalyzer hand_analyzer{hand};
  const FitStatus status = constraint->Fits(hand_analyzer, obs, {});
  SPIEL_CHECK_TRUE(status.IsCertain());
}

void OpeningBidNotMadeConstraintTest() {
  const std::vector<int> deal = ble::Permutation(ble::kNumCards);
  auto state = ConstructStateFromDeal(deal, ble::default_game);
  const auto constraint = LoadConstraint("Can open", {});
  const ble::BridgeMove pass_move = ConstructAuctionMoveFromString("pass");
  for (int num_pass = 0; num_pass < ble::kNumPlayers - 1; ++num_pass) {
    const ble::BridgeObservation obs{state};
    const auto hand = obs.Hands()[0];
    const HandAnalyzer hand_analyzer{hand};
    const FitStatus status = constraint->Fits(hand_analyzer, obs, {});
    SPIEL_CHECK_TRUE(status.IsCertain());
    state.ApplyMove(pass_move);
  }
}

void NoTrumpOpeningConstraintsTest() {
  std::mt19937 rng;
  ble::BridgeHand hand;
  HandAnalyzer hand_analyzer;
  FitStatus status;
  // 1NT (15-17)
  const auto one_nt_constraint = LoadConstraint("1NT",{});
  const std::vector<std::string> one_nt_card_strings = {
      "SA", "S6", "S5", "HK", "HQ", "H2", "D5", "D3", "CA", "CQ", "CJ", "C9",
      "C7"};
  const auto one_nt_state = ConstructStateFromCardStrings({
        {
            one_nt_card_strings,
            {}, {}, {}
        }}, ble::default_game,
    rng);
  const ble::BridgeObservation one_nt_obs{one_nt_state};
  hand = one_nt_obs.Hands()[0];
  hand_analyzer = HandAnalyzer{hand};
  status = one_nt_constraint->Fits(hand_analyzer, one_nt_obs, {});

  SPIEL_CHECK_TRUE(status.IsCertain());

  // 2NT (20-21)
  const auto two_nt_constraint = LoadConstraint("2NT", {});
  // 21 hcp.
  const std::vector<std::string> two_nt_card_strings = {
      "SA", "SK", "S5", "HK", "HQ", "HJ", "D5", "D3", "CA", "CQ", "CJ", "C9",
      "C7"};
  const auto two_nt_state = ConstructStateFromCardStrings({
        {
            two_nt_card_strings,
            {}, {}, {}
        }}, ble::default_game,
    rng);
  const ble::BridgeObservation two_nt_obs{two_nt_state};
  hand = two_nt_obs.Hands()[0];
  hand_analyzer = HandAnalyzer{hand};

  status = two_nt_constraint->Fits(hand_analyzer, two_nt_obs, {});
  SPIEL_CHECK_TRUE(status.IsCertain());

  // 3NT (25-27)
  const auto three_nt_constraint = LoadConstraint("3NT", {});
  // 25 hcp.
  const std::vector<std::string> three_nt_card_strings = {
      "SA", "SK", "S5", "HK", "HQ", "HJ", "DK", "DQ", "CA", "CQ", "CJ", "C9",
      "C7"};
  const auto three_nt_state = ConstructStateFromCardStrings({
        {
            three_nt_card_strings,
            {}, {}, {}
        }}, ble::default_game,
    rng);
  const ble::BridgeObservation three_nt_obs{three_nt_state};
  hand = three_nt_obs.Hands()[0];
  hand_analyzer = HandAnalyzer{hand};
  status = three_nt_constraint->Fits(hand_analyzer, three_nt_obs, {});
  SPIEL_CHECK_TRUE(status.IsCertain());
}

void RuleOf20ConstraintTest() {
  std::mt19937 rng;
  const std::vector<std::string> card_strings = {
      "SK", "SQ", "S5", "S4", "HA", "H8", "H7", "H3", "D6", "CK", "CT", "C6",
      "C4"};
  const auto state = ConstructStateFromCardStrings({
                                                       {
                                                           card_strings,
                                                           {}, {}, {}
                                                       }}, ble::default_game,
                                                   rng);
  const ble::BridgeObservation obs{state};
  const HandAnalyzer hand_analyzer{obs.Hands()[0]};
  const auto rule_of_20_contraint = LoadConstraint("rule_of_20", {});
  const FitStatus status = rule_of_20_contraint->Fits(hand_analyzer, obs, {});
  SPIEL_CHECK_TRUE(status.IsCertain());
}

}

int main(int argc, char* argv[]) {
  sayc::BalancedHandConstraintTest();
  sayc::OpeningBidNotMadeConstraintTest();
  sayc::NoTrumpOpeningConstraintsTest();
  sayc::RuleOf20ConstraintTest();
}