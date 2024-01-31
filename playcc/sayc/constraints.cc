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
                                           std::shared_ptr<Constraint>
                                           constraint) {
  RegisterConstraint(constraint_name, std::move(constraint));
}

std::shared_ptr<Constraint> ConstraintRegisterer::GetByName(
    const std::string& constraint_name) {
  const auto iter = factories().find(constraint_name);
  if (iter == factories().end()) {
    SpielFatalError(absl::StrCat("Unknown bot '",
                                 constraint_name,
                                 "'. Available bots are:\n",
                                 absl::StrJoin(RegisteredConstraints(), "\n")));
  }
  const std::shared_ptr<Constraint>& constraint = iter->second;
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
    std::shared_ptr<Constraint>
    constraint) {
  factories()[constraint_name] = std::move(constraint);
}

bool IsConstraintRegistered(const std::string& constraint_name) {
  return ConstraintRegisterer::IsConstraintRegistered(constraint_name);
}

std::vector<std::string> RegisteredConstraints() {
  return ConstraintRegisterer::RegisteredConstraints();
}

std::shared_ptr<Constraint>
LoadConstraint(const std::string& constraint_name) {
  return ConstraintRegisterer::GetByName(constraint_name);
}

FitStatus BalancedHandConstraint::Fits(const HandAnalyzer& hand_analyzer,
                                       const ble::BridgeObservation& obs)
const {
  if (const bool is_balanced = hand_analyzer.IsBalanced(); !is_balanced) {
    return FitStatus{kImpossible};
  }
  return FitStatus{kCertain};
}

REGISTER_CONSTRAINT("balanced_hand", kBalancedHandConstraint);

FitStatus OpeningBidNotMadeConstraint::Fits(const HandAnalyzer& hand_analyzer,
                                            const ble::BridgeObservation& obs)
const {
  if (HasOpeningBidBeenMade(obs)) {
    return FitStatus{kImpossible};
  } else {
    return FitStatus{kCertain};
  }
}

FitStatus HCPInRangeConstraint::Fits(const HandAnalyzer& hand_analyzer,
                                     const ble::BridgeObservation& obs) const {
  if (range_.Contains(hand_analyzer.HighCardPoints())) {
    return FitStatus{kCertain};
  }
  return FitStatus{kImpossible};
}

FitStatus AndConstraints::Fits(const HandAnalyzer& hand_analyzer,
                               const ble::BridgeObservation& obs) const {
  FitStatus sum{};
  for (const auto& constraint : constraints_) {
    const auto res = constraint->Fits(hand_analyzer, obs);
    if (res == FitStatus(kImpossible)) {
      return FitStatus{kImpossible};
    }
    sum = sum + res;
  }
  return sum;
}

FitStatus OrConstraints::Fits(const HandAnalyzer& hand_analyzer,
                              const ble::BridgeObservation& obs) const {
  FitStatus or_res{};
  for (const auto& constraint : constraints_) {
    const auto res = constraint->Fits(hand_analyzer, obs);
    if (res == FitStatus(kCertain)) {
      return FitStatus{kCertain};
    }
    or_res = or_res || res;
  }
  return or_res;
}

FitStatus RuleOf20Constraint::Fits(const HandAnalyzer& hand_analyzer,
                                   const ble::BridgeObservation& obs) const {
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
}