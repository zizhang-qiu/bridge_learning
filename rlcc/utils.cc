//
// Created by 13738 on 2024/8/28.
//
#include "utils.h"

std::unordered_map<std::string, std::string> ParametersFromString(const std::string &str) {
  std::unordered_map<std::string, std::string> params{};
  if (str.empty())return params;
  int first_paren = str.find('(');
  if (first_paren == std::string::npos) {
    params["name"] = str;
    return params;
  }

  params["name"] = str.substr(0, first_paren);
  int start = first_paren + 1;
  int parens = 1;
  int equals = -1;
  for (int i = start + 1; i < str.length(); ++i) {
    if (str[i] == '(') {
      ++parens;
    } else if (str[i] == ')') {
      --parens;
    } else if (str[i] == '=' && parens == 1) {
      equals = i;
    }
    if ((str[i] == ',' && parens == 1) ||
        (str[i] == ')' && parens == 0 && i > start + 1)) {
      params[str.substr(start, equals - start)] = str.substr(equals + 1, i - equals - 1);
      start = i + 1;
      equals = -1;
    }
  }
  if (parens > 0) {
    std::cerr << "The string " << str << " is missing closing bracket ')'." << std::endl;
    std::abort();
  }
  return params;
}