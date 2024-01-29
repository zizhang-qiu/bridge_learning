//
// Created by qzz on 2024/1/25.
//

#ifndef BRIDGE_LEARNING_PLAYCC_RULE_BASED_DEFENDER_H_
#define BRIDGE_LEARNING_PLAYCC_RULE_BASED_DEFENDER_H_
#include <functional>
#include <map>

#include "bridge_lib/bridge_state.h"

namespace ble = bridge_learning_env;

// https://www.bridgebum.com/bridge_defense.php
enum DefenderRule {
  // Principles.
  kRuleOf10And12,
  kRuleOf11,
  kThirdHandHigh,
  // Opening lead.
  kAceFromAceKing,
  kFalsecardsOnOpeningLead,
  kFourthBestLeads,
  kJackDeniesTenImplies,
  kJournalistLeads,
  kMUD,
  kRusinowLeads,
  kStandardLeads,
  kThirdAndFifthLeads,
  kTrumpLeads,
  // Discard.
  kLavinthalDiscards,
  kOddEvenDiscards,
  kRevolvingDiscards,
  // SignalType
  kAttitudeSignals,
  kBechgaardSignals,
  kCountSignals,
  kFosterEcho,
  kPresentCount,
  kScanianSignals,
  kSequenceSignals,
  kSmithEcho,
  kSuitPreferenceSignals,
  kTrumpEcho,
  kUpsideDownCountAndAttitude,
  kVinjeSignals,
  // Defensive Play
  kCrocodileCoup,
  kDeschapellesCoup,
  kEmperorsCoup,
  kGrosvenorGambit,
  kDuckingPlay,
  kIdiotCoup,
  kMerrimacCoup,
  kTrumpPromotion,
  kUnblockingPlay,
  kUnderruff,
  kUppercut
};

const std::vector<std::string> kRuleStrings = {
    "Rule of 10 and 12",
    "Rule of 11",
    "Third Hand High",
    "Ace from Ace-King",
    "Falsecards on opening lead",
    "Fourth-best leads",
    "Jack denies, Ten implies",
    "Journalist leads",
    "MUD",
    "Rusinow leads",
    "Standard leads",
    "Third and fifth leads",
    "Trump leads",
    "Lavinthal discards",
    "Odd-even discards",
    "Revolving discards",
    "Attitude signals",
    "Bechgaard signals",
    "Count signals",
    "Foster echo",
    "Present count",
    "Scanian signals",
    "Sequence signals",
    "Smith echo",
    "Suit-preference signals",
    "Trump echo",
    "Upside-down count and attitude",
    "Vinje signals",
    "Crocodile Coup",
    "Deschapelles Coup",
    "Emperor's Coup",
    "Grosvenor Gambit",
    "Ducking (Hold-up) Play",
    "Idiot Coup",
    "Merrimac Coup",
    "Trump Promotion",
    "Unblocking Play",
    "Underruff",
    "Uppercut"
};

// enum OpeningLeadRules {};
//
// enum DiscardRules {
//
// };
//
// enum SignalType {
// };
//
// enum DefensivePlays {
//
// };

std::string DefenderRuleToString(DefenderRule defender_rule);

struct RuleResults {
  // Priciples will get how many higher cards declarer holds in the suit.
  int higher_cards_declarer_hold = -1;

};

using RuleFunc = std::function<RuleResults(const ble::BridgeState&)>;

class RuleRegisterer {
  public:
    RuleRegisterer(const std::string& rule_name,
                   const RuleFunc& rule_func);

    static void RegisterRule(const std::string& rule_name,
                             const RuleFunc& rule_func);

    static RuleFunc GetByName(const std::string& rule_name);

    static bool IsRuleRegistered(const std::string& rule_name);

  private:
    static std::map<std::string, RuleFunc>
    factories() {
      static std::map<std::string, RuleFunc> impl;
      return impl;
    }
};

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define REGISTER_DEFENDER_RULE(info, func) RuleRegisterer CONCAT(rule, __COUNTER__)(info, func);

RuleResults RuleOf10And12(const ble::BridgeState& state);

#endif //BRIDGE_LEARNING_PLAYCC_RULE_BASED_DEFENDER_H_
