//
// Created by qzz on 2024/1/29.
//

#include "named_enum.h"

namespace sayc {

NamedEnum::NamedEnum(const std::initializer_list<std::string>& args) {
  int index = 0;
  for (const auto& arg : args) {
    NamedEnumValue value(arg);
    values_.push_back(value);
    value_map_[arg] = index;
    ++index;
  }
}

NamedEnumValue NamedEnum::Get(const std::string& key) const {
  const auto it = value_map_.find(key);
  if (it != value_map_.end()) {
    return values_[it->second];
  }
  SpielFatalError(absl::StrFormat("Key %s not in NamedEnum!", key));

}

const NamedEnumValue& NamedEnum::operator[](const size_t index) const {
  if (index < Size()) {
    return values_[index];
  }
  SpielFatalError(absl::StrFormat("Index %d out of bound, size : %d", index,
                                  Size()));
}
}