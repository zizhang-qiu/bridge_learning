//
// Created by qzz on 2024/1/17.
//

#ifndef MULTISTEP_TRANSITION_BUFFER_H
#define MULTISTEP_TRANSITION_BUFFER_H
#include "future_actor.h"
#include "model.h"
#include "prioritized_replay2.h"
#include "utils.h"

namespace rela {
class MultiStepTransitionBuffer2 {
  public:
    MultiStepTransitionBuffer2(int multiStep,
                               float gamma,
                               bool calcCumulativeReward = false,
                               bool getTrajNextObs = false)
      : multiStep_(multiStep)
        , gamma_(gamma)
        , calcCumulativeReward_(calcCumulativeReward)
        , getTrajNextObs_(getTrajNextObs) {
    }

    void push(TensorDict& dicts) {
      assert((int)history_.size() <= multiStep_);
      tensor_dict::assertKeyExists(dicts, {"reward", "terminal"});

      history_.push_back(dicts);
    }

    size_t size() {
      return history_.size();
    }

    bool canPop() {
      return (int)history_.size() == multiStep_ + 1;
    }

    /* assumes that:
     *  history contains content in t, t+1, ..., t+n
     *  each entry (e.g., "s") has size [1, customized dimension].
     * returns
     *  oldest entry with accumulated reward.
     *  e.g., "s": [multiStep, customized dimension].
     */
    Transition popTransition() {
      assert(canPop());

      TensorDict d = history_.front();

      if (calcCumulativeReward_) {
        // calculate cumulated rewards.
        // history_[multiStep_] is the most recent state.
        torch::Tensor reward = history_[multiStep_]["v"].clone();

        for (int step = multiStep_ - 1; step >= 0; step--) {
          // [TODO] suppose we shouldn't change h.
          auto& h = history_[step];
          const auto& r = h["reward"];

          if (h["terminal"].item<bool>()) {
            // Has terminal. so we reset the reward.
            reward = r;
          }
          else {
            reward = gamma_ * reward + r;
          }
        }

        d["R"] = reward;
      }

      if (getTrajNextObs_) {
        // This is the next obs after multiStep_
        d = tensor_dict::combineTensorDictArgs(d, history_.back());
      }

      history_.pop_front();
      return Transition(d);
    }

    void clear() {
      history_.clear();
    }

  private:
    const int multiStep_;
    const float gamma_;
    const bool calcCumulativeReward_;
    const bool getTrajNextObs_;

    std::deque<TensorDict> history_;
};
}
#endif //MULTISTEP_TRANSITION_BUFFER_H
