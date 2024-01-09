#include "bridge_scoring.h"

#include <algorithm>
#include <string>
#include "bridge_scoring.h"

#include <algorithm>
#include <string>

namespace bridge_learning_env {

std::ostream &operator<<(std::ostream &stream, const Contract &contract) {
  stream << contract.ToString();
  return stream;
}

constexpr int kBaseTrickScores[] = {20, 20, 30, 30, 30};

int ScoreContract(const Contract contract, const DoubleStatus double_status) {
  int score = contract.level * kBaseTrickScores[contract.denomination];
  if (contract.denomination == kNoTrump) score += 10;
  return score * double_status;
}

// Score for failing to make the contract (will be negative).
int ScoreUndertricks(const int undertricks, const bool is_vulnerable, const DoubleStatus double_status) {
  if (double_status == kUndoubled) {
    return (is_vulnerable ? -100 : -50) * undertricks;
  }
  int score = 0;
  if (is_vulnerable) {
    score = -200 - 300 * (undertricks - 1);
  } else {
    if (undertricks == 1) {
      score = -100;
    } else if (undertricks == 2) {
      score = -300;
    } else {
      // This takes into account the -100 for the fourth and subsequent tricks.
      score = -500 - 300 * (undertricks - 3);
    }
  }
  return score * (double_status / 2);
}

// Score for tricks made in excess of the bid.
int ScoreOvertricks(const Denomination trump_suit,
                    const int overtricks,
                    const bool is_vulnerable,
                    const DoubleStatus double_status) {
  if (double_status == kUndoubled) {
    return overtricks * kBaseTrickScores[trump_suit];
  } else {
    return (is_vulnerable ? 100 : 50) * overtricks * double_status;
  }
}

// Bonus for making a doubled or redoubled contract.
int ScoreDoubledBonus(const DoubleStatus double_status) { return 50 * (double_status / 2); }

// Bonuses for partscore, game, or slam.
int ScoreBonuses(const int level, const int contract_score, const bool is_vulnerable) {
  if (level == 7) { // 1500/1000 for grand slam + 500/300 for game
    return is_vulnerable ? 2000 : 1300;
  } else if (level == 6) { // 750/500 for small slam + 500/300 for game
    return is_vulnerable ? 1250 : 800;
  } else if (contract_score >= 100) { // game bonus
    return is_vulnerable ? 500 : 300;
  } else { // partscore bonus
    return 50;
  }
}

int Score(const Contract contract, const int declarer_tricks, const bool is_vulnerable) {
  if (contract.level == 0) return 0;
  const int contracted_tricks = 6 + contract.level;
  const int contract_result = declarer_tricks - contracted_tricks;
  if (contract_result < 0) {
    return ScoreUndertricks(-contract_result, is_vulnerable, contract.double_status);
  } else {
    const int contract_score = ScoreContract(contract, contract.double_status);
    const int bonuses = ScoreBonuses(contract.level, contract_score, is_vulnerable) +
        ScoreDoubledBonus(contract.double_status) +
        ScoreOvertricks(contract.denomination, contract_result, is_vulnerable, contract.double_status);
    return contract_score + bonuses;
  }
}
constexpr int kScoreTable[] = {15, 45, 85, 125, 165, 215, 265, 315, 365, 425, 495, 595,
                               745, 895, 1095, 1295, 1495, 1745, 1995, 2245, 2495, 2995, 3495, 3995};
constexpr int kScoreTableSize = sizeof(kScoreTable) / sizeof(int);
int GetImp(const int score1, const int score2) {
  const int score = score1 - score2;
  const int sign = score == 0 ? 0 : (score > 0 ? 1 : -1);
  const int abs_score = std::abs(score);
  const int p = static_cast<int>(std::upper_bound(kScoreTable, kScoreTable + kScoreTableSize, abs_score) - kScoreTable);
  return sign * p;
}

std::string Contract::ToString() const {
  if (level == 0) return "Passed Out";
  std::string str = std::to_string(level) + std::string{kDenominationChar[denomination]};
  if (double_status == kDoubled) str += "X";
  if (double_status == kRedoubled) str += "XX";
  str += " " + std::string{kPlayerChar[declarer]};
  return str;
}

int Contract::Index() const {
  if (level == 0) return 0;
  int index = level - 1;
  index *= kNumDenominations;
  index += static_cast<int>(denomination);
  index *= kNumPlayers;
  index += static_cast<int>(declarer);
  index *= kNumDoubleStatus;
  if (double_status == kRedoubled) index += 2;
  if (double_status == kDoubled) index += 1;
  return index + 1;
}
} // namespace bridge_learning_env
