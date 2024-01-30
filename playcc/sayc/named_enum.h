//
// Created by qzz on 2024/1/29.
//

#ifndef NAMED_ENUM_H
#define NAMED_ENUM_H
#include <absl/strings/str_format.h>
#include <playcc/log_utils.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace sayc {
class NamedEnumValue {
  public:
    NamedEnumValue(const std::string& key)
      : key_(key) {}

    const std::string& GetKey() const {
      return key_;
    }

  private:
    std::string key_;
};

class NamedEnum {
  public:
    NamedEnum(const std::initializer_list<std::string>& args);

    NamedEnumValue Get(const std::string& key) const;

    size_t Size() const {
      return values_.size();
    }

    const NamedEnumValue& operator[](const size_t index) const;

    const std::vector<NamedEnumValue>& GetValues() const {
      return values_;
    }

  private:
    std::vector<NamedEnumValue> values_;
    std::unordered_map<std::string, int> value_map_;
};
}
#endif //NAMED_ENUM_H
