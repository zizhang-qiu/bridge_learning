#include "utils.h"

#include <algorithm>
#include <random>

namespace bridge_learning_env {
template<>
int ParameterValue<int>(const GameParameters &params, const std::string &key,
                        int default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stoi(iter->second);
}

template<>
std::string ParameterValue<std::string>(const GameParameters &params,
                                        const std::string &key,
                                        std::string default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return iter->second;
}

template<>
double ParameterValue<double>(const GameParameters &params, const std::string &key,
                              double default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return std::stod(iter->second);
}

template<>
bool ParameterValue<bool>(const GameParameters &params, const std::string &key,
                          bool default_value) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }

  return iter->second == "1" || iter->second == "true" ||
      iter->second == "True";
}

std::string GameParametersToString(const GameParameters& params) {
  std::string str{};
  if(params.empty()){return "";}
  if (params.count("name")){str = params.at("name");}
  str.push_back('(');
  bool first = true;
  for(const auto& [key, value] : params) {
    if (key!="name") {
      if(!first) str.push_back(',');
      str.append(key);
      str.append("=");
      str.append(value);
      first = false;
    }
  }
  str.push_back(')');
  return str;
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

std::vector<int> Permutation(int num, std::mt19937 &rng) {
  std::vector<int> ret = Arange(0, num);
  std::shuffle(ret.begin(), ret.end(), rng);
  return ret;
}
std::vector<std::string> StrSplit(const std::string &str, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    result.push_back(token);
  }

  return result;
}
}  // namespace bridge