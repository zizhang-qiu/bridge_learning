//
// Created by qzz on 2024/1/30.
//

#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H
#include <bridge_lib/bridge_observation.h>

#include <map>

#include "hand_analyzer.h"
#include "hand_info.h"

namespace sayc {
enum StatusType {
  kUnknown = -1,
  kImpossible,
  kPossible,
  kCertain,
};

class FitStatus {

  public:
    FitStatus()
      : FitStatus(kUnknown) {}

    explicit FitStatus(const StatusType status_type)
      : status_type_(status_type) {}

    FitStatus operator+(const FitStatus& rhs) const {
      if (status_type_ == kUnknown) {
        return rhs;
      }
      if (rhs.GetStatusType() == kImpossible ||
          GetStatusType() == kImpossible) {
        return FitStatus(kImpossible);
      }
      if (rhs.GetStatusType() == kPossible ||
          GetStatusType() == kPossible) {
        return FitStatus(kPossible);
      }
      return FitStatus(kCertain);
    }

    FitStatus operator||(const FitStatus& rhs) const {
      if (status_type_ == kUnknown) {
        return rhs;
      }
      if (status_type_ == kCertain || rhs.GetStatusType() == kCertain) {
        return FitStatus(kCertain);
      }
      if (status_type_ == kPossible || rhs.GetStatusType() == kPossible) {
        return FitStatus{kPossible};
      }
      return FitStatus{kImpossible};
    }

    bool operator==(const FitStatus& other) const {
      return status_type_ == other.GetStatusType();
    }

    [[nodiscard]] StatusType GetStatusType() const {
      return status_type_;
    }

    friend std::ostream& operator<<(std::ostream& stream,
                                    const FitStatus& status) {
      switch (status.GetStatusType()) {
        case kUnknown:
          stream << "Unknown";
          break;
        case kImpossible:
          stream << "Impossible";
          break;
        case kPossible:
          stream << "Possible";
          break;
        case kCertain:
          stream << "Certain";
          break;
        default:
          SpielFatalError("Unexpected status type.");
      }
      return stream;
    }

  private:
    StatusType status_type_;

};

class Constraint {
  public:
    Constraint() = default;

    virtual ~Constraint() = default;

    [[nodiscard]] virtual FitStatus Fits(const HandAnalyzer& hand_analyzer,
                                         const ble::BridgeObservation& obs)
    const {
      SpielFatalError("Fits function not implemented!");
    }
};

class ConstraintRegisterer {
  public:
    ConstraintRegisterer(const std::string& constraint_name,
                         std::shared_ptr<Constraint> constraint);

    static std::shared_ptr<Constraint> GetByName(
        const std::string& constraint_name);

    static bool IsConstraintRegistered(const std::string& constraint_name);

    static std::vector<std::string> RegisteredConstraints();

    static void RegisterConstraint(const std::string& constraint_name,
                                   std::shared_ptr<Constraint> constraint);

  private:
    static std::map<std::string, std::shared_ptr<Constraint>>& factories() {
      static std::map<std::string, std::shared_ptr<Constraint>> impl;
      return impl;
    }
};

bool IsConstraintRegistered(const std::string& constraint_name);

std::vector<std::string> RegisteredConstraints();

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define REGISTER_CONSTRAINT(info, factory)  \
  ConstraintRegisterer CONCAT(constraint, __COUNTER__)(info, factory);

std::shared_ptr<Constraint> LoadConstraint(const std::string& constraint_name);

class BalancedHandConstraint : public Constraint {
  public:
    BalancedHandConstraint() = default;

    [[nodiscard]] FitStatus Fits(const HandAnalyzer& hand_analyzer,
                                 const ble::BridgeObservation& obs)
    const override;
};

const auto kBalancedHandConstraint = std::make_shared<BalancedHandConstraint>();

class OpeningBidNotMadeConstraint : public Constraint {
  public:
    OpeningBidNotMadeConstraint() = default;

    [[nodiscard]] FitStatus Fits(const HandAnalyzer& hand_analyzer,
                                 const ble::BridgeObservation& obs)
    const override;
};

const auto kOpeningBidNotMadeConstraint = std::make_shared<
  OpeningBidNotMadeConstraint>();

class HCPInRangeConstraint : public Constraint {
  public:
    explicit HCPInRangeConstraint(const Range& range)
      : range_(range) {}

    [[nodiscard]] FitStatus Fits(const HandAnalyzer& hand_analyzer,
                                 const ble::BridgeObservation& obs)
    const override;

  private:
    const Range range_;
};

class AndConstraints : public Constraint {
  public:
    explicit AndConstraints(
        const std::vector<std::shared_ptr<Constraint>>& constraints)
      : constraints_(constraints) { SPIEL_CHECK_FALSE(constraints_.empty()); }

    [[nodiscard]] FitStatus Fits(const HandAnalyzer& hand_analyzer,
                                 const ble::BridgeObservation& obs)
    const override;

  private:
    const std::vector<std::shared_ptr<Constraint>> constraints_;
};

class OrConstraints : public Constraint {
  public:
    explicit OrConstraints(
        const std::vector<std::shared_ptr<Constraint>>& constraints)
      : constraints_(constraints) { SPIEL_CHECK_FALSE(constraints_.empty()); }

    [[nodiscard]] FitStatus Fits(const HandAnalyzer& hand_analyzer,
                                 const ble::BridgeObservation& obs)
    const override;

  private:
    const std::vector<std::shared_ptr<Constraint>> constraints_;
};

const Range kOneNoTrumpOpeningHCPRange{15, 17};
const Range kTwoNoTrumpOpeningHCPRange{20, 21};
const Range kThreeNoTrumpOpeningHCPRange{25, 27};

const auto kOneNoTrumpOpeningHCPConstraint = std::make_shared<
  HCPInRangeConstraint>(kOneNoTrumpOpeningHCPConstraint);
const auto kTwoNoTrumpOpeningHCPConstraint = std::make_shared<
  HCPInRangeConstraint>(kTwoNoTrumpOpeningHCPConstraint);
const auto kThreeNoTrumpOpeningHCPConstraint = std::make_shared<
  HCPInRangeConstraint>(kThreeNoTrumpOpeningHCPConstraint);

// Notrump opening bids are made with balanced hands and may include
// a five-card suit, major or minor.
// 1NT = 15-17 HCP.
const auto OneNoTrumpOpeningConstraint = std::make_shared<AndConstraints>(
{kOpeningBidNotMadeConstraint,
 kOneNoTrumpOpeningHCPConstraint,
 kBalancedHandConstraint});

REGISTER_CONSTRAINT("1NT", OneNoTrumpOpeningConstraint);

const auto TwoNoTrumpOpeningConstraint = std::make_shared<AndConstraints>(
{kOpeningBidNotMadeConstraint,
 kTwoNoTrumpOpeningHCPConstraint,
 kBalancedHandConstraint});

REGISTER_CONSTRAINT("2NT", TwoNoTrumpOpeningConstraint);

const auto ThreeNoTrumpOpeningConstraint = std::make_shared<AndConstraints>(
{kOpeningBidNotMadeConstraint,
 kThreeNoTrumpOpeningHCPConstraint,
 kBalancedHandConstraint});

REGISTER_CONSTRAINT("3NT", ThreeNoTrumpOpeningConstraint);

//
class RuleOf20Constraint : public Constraint {
  RuleOf20Constraint() = default;

  public:
    [[nodiscard]] FitStatus Fits(const HandAnalyzer& hand_analyzer,
                                 const ble::BridgeObservation& obs)
    const override;

  private:
    const int rule_points_ = 20;

};
}
#endif //CONSTRAINTS_H