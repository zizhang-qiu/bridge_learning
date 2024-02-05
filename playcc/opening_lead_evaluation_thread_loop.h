//
// Created by qzz on 2024/1/21.
//

#ifndef OPENING_LEAD_EVALUATION_THREAD_LOOP_H
#define OPENING_LEAD_EVALUATION_THREAD_LOOP_H
#include "dds_evaluator.h"
#include "play_bot.h"
#include "rela/thread_loop.h"
#include "common_utils/threaded_queue.h"

namespace ble = bridge_learning_env;

class OpeningLeadEvaluationThreadLoop : public rela::ThreadLoop {
  public:
    OpeningLeadEvaluationThreadLoop(const std::shared_ptr<DDSEvaluator>& dds_evaluator,
                                    const std::shared_ptr<PlayBot>& bot,
                                    const std::shared_ptr<ble::BridgeGame>&
                                    game,
                                    const std::vector<std::vector<int>>&
                                    trajectories,
                                    ThreadedQueue<int>* bot_evaluation,
                                    const int thread_idx = 0,
                                    const bool verbose = false)
      : dds_evaluator_(dds_evaluator),
        bot_(bot),
        game_(game),
        trajectories_(trajectories),
        bot_evaluation_(bot_evaluation),
        thread_idx_(thread_idx),
        verbose_(verbose) {}

    void mainLoop() override;

  private:
    std::shared_ptr<DDSEvaluator> dds_evaluator_;
    std::shared_ptr<PlayBot> bot_;
    std::shared_ptr<ble::BridgeGame> game_;
    std::vector<std::vector<int>> trajectories_;
    ThreadedQueue<int>* bot_evaluation_;
    const int thread_idx_;
    const bool verbose_;

};
#endif //OPENING_LEAD_EVALUATION_THREAD_LOOP_H
