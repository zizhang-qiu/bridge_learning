//
// Created by qzz on 2024/1/25.
//

#include "rule_based_defender.h"

#include "log_utils.h"
#include "absl/strings/str_cat.h"

std::string DefenderRuleToString(const DefenderRule defender_rule) {
  return kRuleStrings[defender_rule];
}

RuleRegisterer::RuleRegisterer(const std::string& defender_rule,
                               const RuleFunc& rule_func) {
  RegisterRule(defender_rule, rule_func);
}

void RuleRegisterer::RegisterRule(const std::string& defender_rule,
                                  const RuleFunc& rule_func) {
  factories()[defender_rule] = rule_func;
}

RuleFunc RuleRegisterer::GetByName(const std::string& rule_name) {
  const auto it = factories().find(rule_name);
  if (it == factories().end()) {
    SpielFatalError(absl::StrCat("Unregistered rule:", rule_name));
  } else {
    const RuleFunc rule_func = it->second;
    return rule_func;
  }
}

bool RuleRegisterer::IsRuleRegistered(const std::string& rule_name) {
  return factories().count(rule_name);
}

ble::BridgeHistoryItem GetOpeningLead(const ble::BridgeState& state) {
  SPIEL_CHECK_GE(state.NumCardsPlayed(), 1);
  const auto play_history = state.PlayHistory();
  return play_history[0];
}

bool IsThirdHandPlayer(const ble::BridgeState& state,
                       const ble::Player player) {
  SPIEL_CHECK_TRUE(state.IsInPhase(ble::Phase::kPlay));
  const ble::Player declarer = state.GetContract().declarer;
  return (declarer + 3) % ble::kNumPlayers == player;
}

bool IsThirdHandPlayer(const ble::BridgeState& state) {
  return IsThirdHandPlayer(state, state.CurrentPlayer());
}

int NumCardsHigherThanCardInHand(const ble::BridgeHand& hand,
                                 const ble::BridgeMove& card_led) {
  int res = 0;
  for (const auto& card : hand.Cards()) {
    if (card.CardSuit() == card_led.CardSuit()) {
      res += card.Rank() > card_led.CardRank();
    }
  }
  return res;
}

RuleResults RuleOf10And12(const ble::BridgeState& state) {
  const auto opening_lead = GetOpeningLead(state);
  const int opening_lead_rank = opening_lead.rank + 2;

  const auto& dummy_hand = state.DummyHand();
  const ble::Player third_player = (state.GetDummy() + 1) % ble::kNumPlayers;
  const auto& my_hand = state.Hands()[third_player];
  // Subtract the opening lead spot card from 10
  int ans = 10 - opening_lead_rank;
  std::cout << ans << std::endl;
  // Also subtract the number of cards in dummy that are higher than the card led.
  ans -= NumCardsHigherThanCardInHand(dummy_hand, opening_lead.move);
  std::cout << ans << std::endl;
  // Finally, subtract the number of cards in your hand that are higher than the card led.
  ans -= NumCardsHigherThanCardInHand(my_hand, opening_lead.move);
  std::cout << ans << std::endl;
  RuleResults res{};
  res.higher_cards_declarer_hold = ans;
  return res;
}

REGISTER_DEFENDER_RULE(kRuleStrings[DefenderRule::kRuleOf10And12], RuleOf10And12);

RuleResults ThirdAndFifthLeads(const ble::BridgeState& state) {
  return {};
}
