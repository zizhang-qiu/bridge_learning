//
// Created by qzz on 2023/10/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PLAY_BOT_H_
#define BRIDGE_LEARNING_PLAYCC_PLAY_BOT_H_
#include <map>

#include "log_utils.h"
#include "bridge_lib/bridge_state.h"

namespace ble = bridge_learning_env;

class PlayBot {
  public:
    virtual ~PlayBot() = default;

    virtual ble::BridgeMove Step(const ble::BridgeState& state) = 0;

    virtual std::string Name() const { return "Play"; }

    virtual bool IsClonable() const { return false; }

    virtual std::unique_ptr<PlayBot> Clone() {
      SpielFatalError("Clone method not implemented.");
    }

    // Restarts the bot to its initial state, ready to start a new trajectory.
    virtual void Restart() {
    }

    // Configure the bot to be on the given `state` which can be arbitrary.
    // Bot not supporting this feature can raise an error.
    virtual void RestartAt(const ble::BridgeState& state) {
      SpielFatalError("RestartAt(state) not implemented.");
    }
};

class BotFactory {
  public:
    virtual ~BotFactory() = default;

    virtual std::unique_ptr<PlayBot> Create(std::shared_ptr<const ble::BridgeGame> game,
                                            ble::Player player,
                                            const ble::GameParameters& bot_params) = 0;
};

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define REGISTER_PLAY_BOT(info, factory)  BotRegisterer CONCAT(bot, __COUNTER__)(info, std::make_unique<factory>());

class BotRegisterer {
  public:
    BotRegisterer(const std::string& bot_name,
                  std::unique_ptr<BotFactory> factory);

    static std::unique_ptr<PlayBot> CreateByName(const std::string& bot_name,
                                                 std::shared_ptr<const ble::BridgeGame> game,
                                                 ble::Player player_id,
                                                 const ble::GameParameters& params);

    static std::vector<std::string> RegisteredBots();

    static bool IsBotRegistered(const std::string& bot_name);

    static void RegisterBot(const std::string& bot_name,
                            std::unique_ptr<BotFactory> factory);

  private:
    // Returns a "global" map of registrations (i.e. an object that lives from
    // initialization to the end of the program). Note that we do not just use
    // a static data member, as we want the map to be initialized before first
    // use.
    static std::map<std::string, std::unique_ptr<BotFactory>>& factories() {
      static std::map<std::string, std::unique_ptr<BotFactory>> impl;
      return impl;
    }
};

// Returns true if the bot is registered, false otherwise.
bool IsBotRegistered(const std::string& bot_name);

// Returns a list of registered bots' short names.
std::vector<std::string> RegisteredBots();

// Returns a new bot from the specified string, which is the short
// name plus optional parameters, e.g.
// "fixed_action_preference(action_list=0;1;2;3)"
std::unique_ptr<PlayBot> LoadBot(const std::string& bot_name,
                                 const std::shared_ptr<const ble::BridgeGame>& game,
                                 ble::Player player_id);

// Returns a new bot with the specified parameters.
std::unique_ptr<PlayBot> LoadBot(const std::string& bot_name,
                                 const std::shared_ptr<const ble::BridgeGame>& game,
                                 ble::Player player_id,
                                 const ble::GameParameters& bot_params);

#endif // BRIDGE_LEARNING_PLAYCC_PLAY_BOT_H_
