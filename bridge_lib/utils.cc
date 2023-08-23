#include "utils.h"

#include <algorithm>
#include <random>

namespace bridge {
template <>
int ParameterValue<int>(const Parameters& params, const std::string& key,
                        int default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stoi(iter->second);
}

template <>
std::string ParameterValue<std::string>(const Parameters& params,
                                        const std::string& key,
                                        std::string default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return iter->second;
}

template <>
double ParameterValue<double>(const Parameters& params, const std::string& key,
                              double default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stod(iter->second);
}

template <>
bool ParameterValue<bool>(const Parameters& params, const std::string& key,
                          bool default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return (iter->second == "1" || iter->second == "true" ||
                  iter->second == "True"
              ? true
              : false);
}

std::vector<int> Arange(int start, int end) {
  std::vector<int> rv(end - start);
  for (int i = start; i < end; ++i) {
    rv[i - start] = i;
  }
  return rv;
}

std::vector<int> Permutation(int num) {
  std::random_device rd;
  std::mt19937 rng(rd());
  return Permutation(num, rng);
}

std::vector<int> Permutation(int num, std::mt19937& rng) {
  std::vector<int> ret = Arange(0, num);
  std::shuffle(ret.begin(), ret.end(), rng);
  return ret;
}
}  // namespace bridge