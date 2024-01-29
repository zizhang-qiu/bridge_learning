//
// Created by qzz on 2024/1/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_SAYC_RANGE_H_
#define BRIDGE_LEARNING_PLAYCC_SAYC_RANGE_H_

#include <random>

#include "playcc/log_utils.h"

namespace sayc {
// A range class to store items with ranges in [min, max]
// e.g. 1NT can cause hcp Range(15, 17).
class Range {
  public:
    Range(const int max, const int min)
      : max_(max), min_(min) {
      SPIEL_CHECK_GE(max_, min_);
    }

    Range(const int value)
      : max_(value), min_(value) {}

    Range()
      : Range(-1) {}

    int Sample(std::mt19937& rng) const {
      std::uniform_int_distribution<int> dist(min_, max_);
      return dist(rng);
    }

    void SetMax(const int max) { max_ = max; }

    void SetMin(const int min) { min_ = min; }

    bool Contains(const int value) const;

  private:
    int max_;
    int min_;
};

bool IsInRange(const int value, const Range& range);
}

#endif //BRIDGE_LEARNING_PLAYCC_SAYC_RANGE_H_
