//
// Created by qzz on 2023/12/13.
//

#include "transposition_table.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/str_join.h"

std::string TranspositionTable::ToString() const {
  std::string rv{};
  int index = 0;
  for (const auto &kv : table_) {
    absl::StrAppend(&rv,
                    "Element ",
                    index,
                    ":\nstate:\n",
                    kv.first.ToString(),
                    "\nPareto Front:\n",
                    kv.second.ToString());
    ++index;
  }
  return rv;
}
std::string TranspositionTable::Serialize() const {
  std::string rv;
  for (const auto &kv : table_) {
    // For each key-value, serialize the state firstly.
    rv += "state\n";
    rv += kv.first.Serialize() + "\n\n";
    // Then serialize the pareto front.
    rv += "front\n";
    rv += kv.second.Serialize() + "\n\n";
  }
  return rv;
}
TranspositionTable TranspositionTable::Deserialize(const std::string &str, const std::shared_ptr<ble::BridgeGame> &game) {
  TranspositionTable tt{};
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  auto it = std::find(lines.begin(), lines.end(), "state");
//  int num_state = 0;
  while (true) {
//    std::cout << num_state << std::endl;

    auto next_it = std::find(it + 1, lines.end(), "state");
//    std::vector<std::string> this_kv(it, next_it);
//    std::cout << absl::StrJoin(this_kv, "\n") << std::endl;

    auto front_it = std::find(it + 1, next_it, "front");
    std::vector<std::string> state_lines(it + 1, front_it - 1);
    auto state = ble::BridgeStateWithoutHiddenInfo::Deserialize(absl::StrJoin(state_lines, "\n"), game);
    std::vector<std::string> front_lines(front_it + 1, next_it - 1);
    auto front = ParetoFront::Deserialize(absl::StrJoin(front_lines, "\n"));
    tt[state] = front;
    it = std::find(it + 1, lines.end(), "state");
    if (it == lines.end()) {
      break;
    }
//    ++num_state;
  }
  return tt;
}

std::ostream &operator<<(std::ostream &stream, const TranspositionTable &tt) {
  return stream << tt.ToString();
}
bool operator==(const TranspositionTable &lhs, const TranspositionTable &rhs) {
  return lhs.Table() == rhs.Table();
}
