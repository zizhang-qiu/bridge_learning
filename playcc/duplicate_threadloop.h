//
// Created by qzz on 2023/12/29.
//

#ifndef DUPLICATE_THREADLOOP_H
#define DUPLICATE_THREADLOOP_H

#include "rela/thread_loop.h"

#include "play_bot.h"
#include "resampler.h"
#include "threaded_queue.h"

struct EvaluationConfig {
  int num_deals{};
  ble::Contract contract{3, ble::kNoTrump, ble::kUndoubled, ble::kSouth};
};

enum EvaluationResult { kSameResults = 0, kPlayer1Win, kPlayer2Win };

class DuplicateThreadloop final : public rela::ThreadLoop {
  public:
  DuplicateThreadloop(const EvaluationConfig &cfg,
                      const std::shared_ptr<Resampler> &resampler,
                      const std::shared_ptr<PlayBot> &player1,
                      const std::shared_ptr<PlayBot> &player2,
                      const std::shared_ptr<PlayBot> &defender,
                      std::shared_ptr<ThreadedQueue<EvaluationResult>> &queue,
                      const bool verbose = false) :
      cfg_(cfg),
      resampler_(resampler),
      player1_(player1),
      player2_(player2),
      defender_(defender),
      queue_(queue),
      verbose_(verbose) {}

  void mainLoop() override;

  void RunOnce(ble::BridgeState &state, std::shared_ptr<PlayBot> &declarer, std::shared_ptr<PlayBot> &defender) const;

  private:
  const EvaluationConfig cfg_;
  int num_deals_played = 0;
  std::shared_ptr<Resampler> resampler_;
  std::shared_ptr<PlayBot> player1_;
  std::shared_ptr<PlayBot> player2_;
  std::shared_ptr<PlayBot> defender_;
  std::shared_ptr<ThreadedQueue<EvaluationResult>> &queue_;
  const bool verbose_;
};

#endif // DUPLICATE_THREADLOOP_H
