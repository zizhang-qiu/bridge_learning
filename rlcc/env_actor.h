#ifndef RLCC_ENV_ACTOR_H
#define RLCC_ENV_ACTOR_H
#include <memory>
#include <string>
#include "rela/logging.h"
#include "rlcc/bridge_actor.h"
#include "rlcc/env.h"

namespace rlcc {

struct EnvActorOptions {
  bool eval = false;
};

class EnvActor {
 public:
  EnvActor(std::vector<std::shared_ptr<Actor>> actors,
           const EnvActorOptions& options)
      : actors_(actors), options_(options) {}

  virtual void ObserveBeforeAct() {}

  virtual void Act() {}

  virtual void ObserveAfterAct() {}

  virtual void SendExperience() {}

  virtual void PostSendExperience() {}

  void CheckValid(const rlcc::GameEnv& env) const {
    RELA_CHECK_EQ(env.Spec().num_players, actors_.size());
  }

  const std::vector<std::vector<float>>& HistoryRewards() const {
    return history_rewards_;
  }

  const std::vector<std::string>& HistoryInfo() const {
    return history_info_;
  }

  int TerminalCount() const { return terminal_count_; }

  virtual const std::shared_ptr<GameEnv>& GetEnv() const = 0;

 protected:
  void EnvTerminate(const std::vector<float>* rewards = nullptr,
                    const std::string& info = "") {
    ++terminal_count_;

    if (rewards != nullptr && options_.eval) {
      history_rewards_.push_back(*rewards);
      history_info_.push_back(info);
    }
  }

  int terminal_count_ = 0;
  std::vector<std::shared_ptr<Actor>> actors_;
  // Reward of each player in each game.
  std::vector<std::vector<float>> history_rewards_;
  std::vector<std::string> history_info_;
  const EnvActorOptions options_;
};
}  // namespace rlcc
#endif /* RLCC_ENV_ACTOR_H */
