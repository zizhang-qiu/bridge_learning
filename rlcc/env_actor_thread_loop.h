#ifndef RLCC_ENV_ACTOR_THREAD_LOOP_H
#define RLCC_ENV_ACTOR_THREAD_LOOP_H
#include "rela/thread_loop.h"
#include "rlcc/env_actor.h"

namespace rlcc {
class EnvActorThreadLoop : public rela::ThreadLoop {
 public:
  EnvActorThreadLoop(std::vector<std::shared_ptr<EnvActor>> env_actors,
                     int num_game_per_env = -1, int thread_idx = -1,
                     bool verbose = false)
      : thread_idx_(thread_idx),
        env_actors_(env_actors),
        num_game_per_env_(num_game_per_env),
        verbose_(verbose) {}

  void mainLoop() override {
    while (!terminated()) {
      if (paused()) {
        waitUntilResume();
      }

       if (verbose_) {
         std::cout << "Before observe before act" << std::endl;
         std::cout << "Terminal count: " << env_actors_[0]->TerminalCount()
                   << std::endl;
         std::cout << "Current Env:\n"
                   << env_actors_[0]->GetEnv()->ToString() << std::endl;
//         std::cout << "Current Env:\n"
//                   << env_actors_[1]->GetEnv()->ToString() << std::endl;
       }

      for (auto& ea : env_actors_) {
        if (!EnvFinished(*ea)) {
          ea->ObserveBeforeAct();
        }
      }

       if (verbose_) {
         std::cout << "Before act" << std::endl;
       }

      for (auto& ea : env_actors_) {
        if (!EnvFinished(*ea)) {
          ea->Act();
        }
      }
      if(verbose_)
       std::cout << "Before observe after act" << std::endl;
      for (auto& ea : env_actors_) {
        if (!EnvFinished(*ea)) {
          ea->ObserveAfterAct();
        }
      }
      if(verbose_)
       std::cout << "Before send experience" << std::endl;
      for (auto& ea : env_actors_) {
        if (!EnvFinished(*ea)) {
          ea->SendExperience();
        }
      }
      if(verbose_)
       std::cout << "Before post send experience" << std::endl;
      for (auto& ea : env_actors_) {
        if (!EnvFinished(*ea)) {
          ea->PostSendExperience();
        }
      }

      bool all_finished = true;
      for (auto& ea : env_actors_) {
        if (!EnvFinished(*ea)) {
          all_finished = false;
          break;
        }
      }

      if (all_finished) {
        break;
      }

      loop_count_++;
      if (verbose_) {
        std::cout << "Thread " << thread_idx_ << ", Loop count: " << loop_count_
                  << std::endl;
      }
    }
  }

 private:
  const int thread_idx_;
  std::vector<std::shared_ptr<EnvActor>> env_actors_;
  int num_game_per_env_;

  int loop_count_ = 0;
  bool verbose_;

  //Check is an env actor is finished, useful when eval.
  bool EnvFinished(const EnvActor& ea) {
    return (num_game_per_env_ > 0 && ea.TerminalCount() >= num_game_per_env_);
  }
};
}  // namespace rlcc

#endif /* RLCC_ENV_ACTOR_THREAD_LOOP_H */
