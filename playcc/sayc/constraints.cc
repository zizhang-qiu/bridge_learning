//
// Created by qzz on 2024/1/30.
//

#include "constraints.h"

#include <utility>

#include "utils.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace sayc {
ConstraintRegisterer::ConstraintRegisterer(const std::string& constraint_name,
                                           ConstraintFactory factory) {
  RegisterConstraint(constraint_name, std::move(factory));
}

std::shared_ptr<Constraint> ConstraintRegisterer::CreateByName(
    const std::string& constraint_name,
    const ble::GameParameters& params) {
  const auto iter = factories().find(constraint_name);
  if (iter == factories().end()) {
    SpielFatalError(absl::StrCat("Unknown constraint '",
                                 constraint_name,
                                 "'. Available constraints are:\n",
                                 absl::StrJoin(RegisteredConstraints(), "\n")));
  }
  const std::shared_ptr<Constraint>& constraint = iter->second(params);
  return constraint;
}

bool ConstraintRegisterer::IsConstraintRegistered(
    const std::string& constraint_name) {
  return factories().find(constraint_name) != factories().end();
}

std::vector<std::string> ConstraintRegisterer::RegisteredConstraints() {
  std::vector<std::string> names;
  for (const auto& key_val : factories()) {
    names.push_back(key_val.first);
  }
  return names;
}

void ConstraintRegisterer::RegisterConstraint(
    const std::string& constraint_name,
    ConstraintFactory factory) {
  factories()[constraint_name] = std::move(factory);
}

bool IsConstraintRegistered(const std::string& constraint_name) {
  return ConstraintRegisterer::IsConstraintRegistered(constraint_name);
}

std::vector<std::string> RegisteredConstraints() {
  return ConstraintRegisterer::RegisteredConstraints();
}

std::shared_ptr<Constraint>
LoadConstraint(const std::string& constraint_name,
               const ble::GameParameters& params = {}) {
  return ConstraintRegisterer::CreateByName(constraint_name, params);
}

FitStatus BalancedHandConstraint::Fits(const HandAnalyzer& hand_analyzer,
                                       const ble::BridgeObservation& obs,
                                       const std::array<
                                         HandInfo, ble::kNumPlayers - 1>&
                                       hand_infos)
const {
  if (const bool is_balanced = hand_analyzer.IsBalanced(); !is_balanced) {
    return FitStatus{kImpossible};
  }
  return FitStatus{kCertain};
}

REGISTER_CONSTRAINT("balanced_hand",
                    [](const ble::GameParameters& params){return std::
                    make_shared<BalancedHandConstraint>();});

FitStatus OpeningBidNotMadeConstraint::Fits(const HandAnalyzer& hand_analyzer,
                                            const ble::BridgeObservation& obs,
                                            const std::array<
                                              HandInfo, ble::kNumPlayers - 1>&
                                            hand_infos)
const {
  if (HasOpeningBidBeenMade(obs)) {
    return FitStatus{kImpossible};
  }
  return FitStatus{kCertain};

}

const auto kOpeningBidNotMadeConstraint = std::make_shared<
  OpeningBidNotMadeConstraint>();

REGISTER_CONSTRAINT("can_open",
                    [](const ble::GameParameters& params){
                    return std::make_shared<OpeningBidNotMadeConstraint>();});

FitStatus HCPInRangeConstraint::Fits(const HandAnalyzer& hand_analyzer,
                                     const ble::BridgeObservation& obs,
                                     const std::array<
                                       HandInfo, ble::kNumPlayers - 1>&
                                     hand_infos) const {
  if (range_.Contains(hand_analyzer.HighCardPoints())) {
    return FitStatus{kCertain};
  }
  return FitStatus{kImpossible};
}

const Range kOneNoTrumpOpeningHCPRange{15, 17};
const Range kTwoNoTrumpOpeningHCPRange{20, 21};
const Range kThreeNoTrumpOpeningHCPRange{25, 27};

// const auto kOneNoTrumpOpeningHCPConstraint = std::make_shared<
//   HCPInRangeConstraint>(kOneNoTrumpOpeningHCPRange);
// const auto kTwoNoTrumpOpeningHCPConstraint = std::make_shared<
//   HCPInRangeConstraint>(kTwoNoTrumpOpeningHCPRange);
// const auto kThreeNoTrumpOpeningHCPConstraint = std::make_shared<
//   HCPInRangeConstraint>(kThreeNoTrumpOpeningHCPRange);

const auto kOneLevelOpeningHCPConstraint = std::make_shared<
  HCPInRangeConstraint>(Range(0, 21));

//
REGISTER_CONSTRAINT("one_level_open_hcp",
                    [](const ble::GameParameters&){return
                    kOneLevelOpeningHCPConstraint;});

std::shared_ptr<HCPInRangeConstraint> HCPInRangeConstraintFactory(
    const ble::GameParameters& params) {
  const int range_min = ble::ParameterValue<int>(params, "min", 0);
  const int range_max = ble::ParameterValue<int>(params, "max", 0);
  return std::make_shared<HCPInRangeConstraint>(Range{range_min, range_max});
}

REGISTER_CONSTRAINT("hcp_in_range", HCPInRangeConstraintFactory);

const std::shared_ptr<Constraint> kOneNoTrumpOpeningHCPConstraint =
    LoadConstraint(
        "hcp_in_range", {{"min", "15"}, {"max", "17"}});

const std::shared_ptr<Constraint> kTwoNoTrumpOpeningHCPConstraint =
    LoadConstraint(
        "hcp_in_range", {{"min", "20"}, {"max", "21"}});

const std::shared_ptr<Constraint> kThreeNoTrumpOpeningHCPConstraint =
    LoadConstraint(
        "hcp_in_range", {{"min", "25"}, {"max", "27"}});

FitStatus AndConstraints::Fits(const HandAnalyzer& hand_analyzer,
                               const ble::BridgeObservation& obs,
                               const std::array<HandInfo, ble::kNumPlayers - 1>&
                               hand_infos) const {
  FitStatus sum{};
  for (const auto& constraint : constraints_) {
    const auto res = constraint->Fits(hand_analyzer, obs, {});
    if (res == FitStatus(kImpossible)) {
      return FitStatus{kImpossible};
    }
    sum = sum + res;
  }
  return sum;
}

FitStatus OrConstraints::Fits(const HandAnalyzer& hand_analyzer,
                              const ble::BridgeObservation& obs,
                              const std::array<HandInfo, ble::kNumPlayers - 1>&
                              hand_infos) const {
  FitStatus or_res{};
  for (const auto& constraint : constraints_) {
    const auto res = constraint->Fits(hand_analyzer, obs, {});
    if (res == FitStatus(kCertain)) {
      return FitStatus{kCertain};
    }
    or_res = or_res || res;
  }
  return or_res;
}

const auto OneNoTrumpOpeningConstraint = AndConstraints(
{kOpeningBidNotMadeConstraint, kOneNoTrumpOpeningHCPConstraint,
 std::make_shared<BalancedHandConstraint>()});

REGISTER_CONSTRAINT(
    "1NT",
    [](const ble::GameParameters&){return std::make_shared<AndConstraints>(
      OneNoTrumpOpeningConstraint);});

const auto TwoNoTrumpOpeningConstraint = AndConstraints(
{kOpeningBidNotMadeConstraint, kTwoNoTrumpOpeningHCPConstraint,
 std::make_shared<BalancedHandConstraint>()});

REGISTER_CONSTRAINT(
    "2NT",
    [](const ble::GameParameters&){return std::make_shared<AndConstraints>(
      TwoNoTrumpOpeningConstraint);});

const auto ThreeNoTrumpOpeningConstraint = AndConstraints(
{kOpeningBidNotMadeConstraint, kThreeNoTrumpOpeningHCPConstraint,
 std::make_shared<BalancedHandConstraint>()});

REGISTER_CONSTRAINT(
    "3NT",
    [](const ble::GameParameters&){return std::make_shared<AndConstraints>(
      ThreeNoTrumpOpeningConstraint);});

FitStatus RuleOf20Constraint::Fits(const HandAnalyzer& hand_analyzer,
                                   const ble::BridgeObservation& obs,
                                   const std::array<
                                     HandInfo, ble::kNumPlayers - 1>&
                                   hand_infos) const {
  // Take your high card points.
  const int hcp = hand_analyzer.HighCardPoints();
  // Add the number of cards in your longest suit.
  const auto sorted_suit_length = hand_analyzer.GetSortedSuitLength();
  const int num_cards_of_longest_suit = sorted_suit_length[0];
  // Add the number of cards in your second longest suit.
  const int num_cards_of_second_longest_suit = sorted_suit_length[1];

  const int total_points = hcp + num_cards_of_longest_suit +
                           num_cards_of_second_longest_suit;
  if (total_points >= rule_points_) {
    return FitStatus(kCertain);
  }
  return FitStatus(kImpossible);
}

REGISTER_CONSTRAINT("rule_of_20",
                    [](const ble::GameParameters&){return std::
                    make_shared<RuleOf20Constraint>();});

FitStatus RuleOf15Constraint::Fits(const HandAnalyzer& hand_analyzer,
                                   const ble::BridgeObservation& obs,
                                   const std::array<
                                     HandInfo, ble::kNumPlayers - 1>&
                                   hand_infos) const {
  // Take your HCP and add the number of spades you hold. If the total is 15 or more, open.
  const int hcp = hand_analyzer.HighCardPoints();

  const int spades_length = hand_analyzer.GetSuitLength()[
    ble::Suit::kSpadesSuit];

  const int total_points = hcp + spades_length;
  if (total_points >= rule_points_) {
    return FitStatus(kCertain);
  }
  return FitStatus(kImpossible);
}

REGISTER_CONSTRAINT("rule_of_15",
                    [](const ble::GameParameters&){return std::make_shared<
                    RuleOf15Constraint>();});

FitStatus WeakTwoConstraints::Fits(const HandAnalyzer& hand_analyzer,
                                   const ble::BridgeObservation& obs,
                                   const std::array<
                                     HandInfo, ble::kNumPlayers - 1>&
                                   hand_infos) const {
  const auto sorted_suit_length = hand_analyzer.GetSortedSuitLength();
  // A six-card suit.
  const bool six_card_suit = sorted_suit_length[0] == 6;
  if (!six_card_suit) {
    return FitStatus(kImpossible);
  }
  // No void.
  const auto suit_length = hand_analyzer.GetSuitLength();
  const bool no_void = std::find(suit_length.begin(), suit_length.end(), 0) !=
                       suit_length.end();
  if (!no_void) {
    return FitStatus(kImpossible);
  }
  // Decent cards in the suit.
  const auto sorted_suit_length_with_suit = hand_analyzer.
      GetSortedSuitLengthWithSuits();
  // At most, we may have two 6-card suits, we need to check both of them.
  const auto [_, six_card_suits] = sorted_suit_length_with_suit[0];
  const int queen_rank = std::find(std::begin(ble::kRankChar),
                                   std::end(ble::kRankChar), 'Q') - std::begin(
                             ble::kRankChar);
  bool is_decent = false;
  for (const ble::Suit suit : six_card_suits) {
    // We use "two of the top three honors" to determine a decent suit.
    int num_match_cards = 0;
    for (const auto& card : hand_analyzer.Hand().CardsBySuits()[suit]) {
      if (card.Rank() >= queen_rank) {
        ++num_match_cards;
      }
    }
    if (num_match_cards >= 2) {
      is_decent = true;
      break;
    }
  }
  if (is_decent) { return FitStatus(kCertain); }
  return FitStatus(kImpossible);
}

}