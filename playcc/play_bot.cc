//
// Created by qzz on 2024/1/6.
//
#include "play_bot.h"

#include "absl/strings/str_join.h"

#include "log_utils.h"

BotRegisterer::BotRegisterer(const std::string& bot_name, std::unique_ptr<BotFactory> factory) {
  RegisterBot(bot_name, std::move(factory));
}

std::unique_ptr<PlayBot> BotRegisterer::CreateByName(const std::string& bot_name,
                                                     std::shared_ptr<const ble::BridgeGame> game,
                                                     ble::Player player_id,
                                                     const ble::GameParameters& params) {
  auto iter = factories().find(bot_name);
  if (iter == factories().end()) {
    SpielFatalError(absl::StrCat("Unknown bot '",
                                 bot_name,
                                 "'. Available bots are:\n",
                                 absl::StrJoin(RegisteredBots(), "\n")));
  }
  else {
    const std::unique_ptr<BotFactory>& factory = iter->second;
    return factory->Create(std::move(game), player_id, params);
  }
}

std::vector<std::string> BotRegisterer::RegisteredBots() {
  std::vector<std::string> names;
  for (const auto& key_val : factories()) names.push_back(key_val.first);
  return names;
}

bool BotRegisterer::IsBotRegistered(const std::string& bot_name) {
  return factories().find(bot_name) != factories().end();
}

void BotRegisterer::RegisterBot(const std::string& bot_name, std::unique_ptr<BotFactory> factory) {
  factories()[bot_name] = std::move(factory);
}

bool IsBotRegistered(const std::string& bot_name) { return BotRegisterer::IsBotRegistered(bot_name); }

std::vector<std::string> RegisteredBots() { return BotRegisterer::RegisteredBots(); }

ble::GameParameters ParametersFromString(const std::string& str) {
  ble::GameParameters params{};
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
    }
    else if (str[i] == ')') {
      --parens;
    }
    else if (str[i] == '=' && parens == 1) {
      equals = i;
    }
    if ((str[i] == ',' && parens == 1) ||
      (str[i] == ')' && parens == 0 && i > start + 1)) {
      params[str.substr(start, equals - start)] = str.substr(equals + 1, i - equals - 1);
      start = i + 1;
      equals = -1;
    }
  }
  if (parens > 0) { SpielFatalError("Missing closing bracket ')'."); }
  return params;
}

std::unique_ptr<PlayBot> LoadBot(const std::string& bot_name,
                                 const std::shared_ptr<const ble::BridgeGame>& game,
                                 ble::Player player_id) {
  const ble::GameParameters params = ParametersFromString(bot_name);
  return LoadBot(params.at("name"), game, player_id, params);
}

std::unique_ptr<PlayBot> LoadBot(const std::string& bot_name,
                                 const std::shared_ptr<const ble::BridgeGame>& game,
                                 ble::Player player_id,
                                 const ble::GameParameters& bot_params) {
  std::unique_ptr<PlayBot> result =
      BotRegisterer::CreateByName(bot_name, game, player_id, bot_params);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat("Unable to create bot: ", bot_name));
  }
  return result;
}
