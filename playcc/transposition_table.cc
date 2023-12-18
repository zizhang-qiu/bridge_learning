//
// Created by qzz on 2023/12/13.
//

#include "transposition_table.h"

#include "absl/strings/str_cat.h"
std::string TranspositionTable::ToString() const {
  std::string rv{};
  int index = 0;
  for (const auto &kv : table_) {
    absl::StrAppend(&rv,
                    "Element ",
                    index,
                    ":\nstate:\n",
                    kv.first.ToString(),
                    "Pareto Front:\n",
                    kv.second.ToString());
    ++index;
  }
  return rv;
}
std::ostream &operator<<(ostream &stream, const TranspositionTable &tt) {
  return stream << tt.ToString();
}
