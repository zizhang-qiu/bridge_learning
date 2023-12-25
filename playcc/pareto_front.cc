//
// Created by qzz on 2023/11/5.
//
#include <algorithm>

#include "pareto_front.h"
#include "absl/strings/str_split.h"
#include "log_utils.h"

ParetoFront::ParetoFront(const std::vector<OutcomeVector> &outcome_vectors) {
  if (outcome_vectors.empty()) {
    outcome_vectors_ = outcome_vectors;
  }
  const size_t size = outcome_vectors.size();
  // For each outcome vector, if another vector dominates it, it should be eliminated.
  for (size_t i = 0; i < size; ++i) {
    bool is_dominated = false;
    bool has_same = false;
    for (size_t j = 0; j < size; ++j) {
      if (OutcomeVectorDominate(outcome_vectors[j], outcome_vectors[i])) {
        is_dominated = true;
        break;
      }
    }
    for (const auto &ov : outcome_vectors_) {
      if (ov == outcome_vectors[i]) {
        has_same = true;
        break;
      }
    }
    if (!is_dominated && !has_same) {
      outcome_vectors_.push_back(outcome_vectors[i]);
    }
  }
}

bool ParetoFront::Insert(const OutcomeVector &outcome_vector) {
  if (outcome_vectors_.empty()) {
    outcome_vectors_.push_back(outcome_vector);
    return true;
  }

  for (auto &ov : outcome_vectors_) {
    if (OutcomeVectorDominate(ov, outcome_vector)) {
      return false;
    }
  }

  const size_t original_size = outcome_vectors_.size();

  outcome_vectors_.erase(
      std::remove_if(outcome_vectors_.begin(),
                     outcome_vectors_.end(),
                     [outcome_vector](const OutcomeVector &vec) { return OutcomeVectorDominate(outcome_vector, vec); }),
      outcome_vectors_.end());

  const bool is_one_dominated = outcome_vectors_.size() < original_size;
  if (is_one_dominated) {
    outcome_vectors_.push_back(outcome_vector);
    return true;
  }

  const bool has_same_vector = HasSameVector(outcome_vector);
  if (!has_same_vector) {
    outcome_vectors_.push_back(outcome_vector);
    return true;
  }
  return false;
}

std::string ParetoFront::ToString() const {
  std::string rv;
  int index = 0;
  for (auto it = outcome_vectors_.begin(); it != outcome_vectors_.end(); ++it) {
    rv += std::to_string(index) + ":\nGame status:\n";
    rv += "[" + VectorToString((*it).game_status) + "],\npossible worlds:\n";
    rv += "[" + VectorToString((*it).possible_world) + "],\n move:";
    rv += (*it).move.ToString();
    ++index;
    if (std::next(it) != outcome_vectors_.end()) {
      rv += ",\n";
    }
  }
  //  rv += "}\npossible worlds:\n";
  //  if (!outcome_vectors_.empty()) {
  //    rv += "[" + VectorToString(outcome_vectors_[0].possible_world) + "]";
  //  }
  return rv;
}
ParetoFront ParetoFront::ParetoFrontWithOneOutcomeVector(const std::vector<int> &possible_worlds,
                                                         const int fill_value) {
  const std::vector<int> game_status(possible_worlds.size(), fill_value);
  const OutcomeVector outcome_vector{game_status, possible_worlds};
  const std::vector<OutcomeVector> outcome_vectors = {outcome_vector};
  return ParetoFront(outcome_vectors);
}
bool ParetoFront::HasSameVector(const OutcomeVector &outcome_vector) const {
  for (const auto &ov : outcome_vectors_) {
    if (ov == outcome_vector) {
      return true;
    }
  }
  return false;
}

double ParetoFront::Score() const {
  double max_score = 0;
  for (const auto &ov : outcome_vectors_) {
    max_score = std::max(ov.Score(), max_score);
  }
  return max_score;
}

OutcomeVector ParetoFront::BestOutcome() const {
  double max_score = -1;
  OutcomeVector result{};
  for (const auto &ov : outcome_vectors_) {
    if (double this_score = ov.Score(); this_score > max_score) {
      result = ov;
      max_score = this_score;
    }
  }
  return result;
}
void ParetoFront::SetMove(const ble::BridgeMove &move) {
  for (auto &ov : outcome_vectors_) {
    ov.move = move;
  }
}

std::string ParetoFront::Serialize() const {
  std::string rv{};
  for (const auto &ov : outcome_vectors_) {
    rv += "game status\n";
    for (const auto &status : ov.game_status) {
      rv += std::to_string(status) + "\n";
    }

    rv += "possible worlds\n";
    for (const auto &status : ov.possible_world) {
      rv += std::to_string(status) + "\n";
    }

    rv += "move\n";
    rv += ble::CardString(ov.move.CardSuit(), ov.move.CardRank()) + "\n";
    rv += "\n";
  }
  return rv;
}

ParetoFront ParetoFront::Deserialize(const std::string &str) {
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  ParetoFront front{};
  auto it = std::find(lines.begin(), lines.end(), "game status");
  if (it==lines.end()){
    return front;
  }
  while (true) {
    auto next_it = std::find(it + 1, lines.end(), "game status");
    std::vector<int> game_status;
    std::vector<int> possible_worlds;
    ble::BridgeMove move{};
    auto possible_worlds_it = std::find(it, next_it, "possible worlds");
    auto move_it = std::find(it, next_it, "move");
    for (auto temp_it = it + 1; temp_it != possible_worlds_it; ++temp_it) {
      if (temp_it->empty()) continue;
      game_status.push_back(std::stoi(*temp_it));
    }

    for (auto temp_it = possible_worlds_it + 1; temp_it != move_it; ++temp_it) {
      if (temp_it->empty()) continue;
      possible_worlds.push_back(std::stoi(*temp_it));
    }

    SPIEL_CHECK_EQ(game_status.size(), possible_worlds.size());

    auto move_str = *(move_it + 1);
    if (move_str != "II") {
      const ble::Suit suit = ble::SuitCharToSuit(move_str[0]);
      const int rank = ble::RankCharToRank(move_str[1]);
      move = ble::BridgeMove{
          /*move_type=*/ble::BridgeMove::kPlay,
          /*suit=*/suit,
          /*rank=*/rank,
          /*denomination=*/ble::kInvalidDenomination,
          /*level=*/-1,
          /*other_call=*/ble::kNotOtherCall
      };
    }
    const OutcomeVector ov{game_status, possible_worlds, move};
    front.Insert(ov);
    it = std::find(it + 1, lines.end(), "game status");
    if (it == lines.end()) {
      break;
    }
  }
  return front;
}

ParetoFront operator*(const ParetoFront &lhs, const ParetoFront &rhs) {
  ParetoFront res{};
  for (const auto &ov_l : lhs.OutcomeVectors()) {
    for (const auto &ov_r : rhs.OutcomeVectors()) {
      auto product = VectorProduct(ov_l.game_status, ov_r.game_status);
      // std::cout << VectorToString(product) << std::endl;
      const OutcomeVector outcome_vector{product, ov_l.possible_world};
      res.Insert(outcome_vector);
    }
  }
  return res;
}

bool operator<=(const ParetoFront &lhs, const ParetoFront &rhs) {
  for (const auto &vec : lhs.OutcomeVectors()) {
    bool one_greater_or_equal = false;
    for (const auto &v : rhs.OutcomeVectors()) {
      if (VectorGreaterEqual(v.game_status, vec.game_status)) {
        one_greater_or_equal = true;
        break;
      }
    }
    if (!one_greater_or_equal) {
      return false;
    }
  }
  return true;
}

ParetoFront ParetoFrontMin(const ParetoFront &lhs, const ParetoFront &rhs) {
  if (lhs.Empty()) {
    return rhs;
  }
  ParetoFront result{};
  for (const auto &vec : lhs.OutcomeVectors()) {
    for (const auto &v : rhs.OutcomeVectors()) {
      //      const auto r_vec = VectorMin(vec.game_status, v.game_status);
      const OutcomeVector outcome_vector = OutcomeVectorJoin(vec, v);
      result.Insert(outcome_vector);
    }
  }
  return result;
}

ParetoFront ParetoFrontMax(const ParetoFront &lhs, const ParetoFront &rhs) {
  if (lhs.Empty()) {
    return rhs;
  }
  ParetoFront result(lhs);
  for (const auto &ov : rhs.OutcomeVectors()) {
    result.Insert(ov);
  }
  return result;
}
std::ostream &operator<<(std::ostream &stream, const ParetoFront &front) {
  stream << front.ToString();
  return stream;
}

// A Pareto front P1 dominates or is equal to a Pareto front P2 iff \forall v \in P2,
// \exist v' \in P1 such that (v' dominates v) or v' = v.
bool ParetoFrontDominate(const ParetoFront &lhs, const ParetoFront &rhs) {
  if (lhs.Empty()) return false;
  for (const OutcomeVector &vec : rhs.OutcomeVectors()) {
    bool one_dominate = false;
    for (const OutcomeVector &v : lhs.OutcomeVectors()) {
      if (OutcomeVectorDominate(v, vec)) {
        one_dominate = true;
        break;
      }
    }
    if (!one_dominate) {
      return false;
    }
  }
  return true;
}

bool operator==(const ParetoFront &lhs, const ParetoFront &rhs) {
  for (const OutcomeVector &vec : rhs.OutcomeVectors()) {
    bool one_equal = false;
    for (const OutcomeVector &v : lhs.OutcomeVectors()) {
      if (v == vec) {
        one_equal = true;
        break;
      }
    }
    if (!one_equal) {
      return false;
    }
  }
  return true;
}
