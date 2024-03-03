//
// Created by qzz on 2024/1/21.
//

#ifndef DDS_EVALUATOR_H
#define DDS_EVALUATOR_H

#include <mutex>
#include <condition_variable>

#include "bridge_lib/third_party/dds/include/dll.h"
#include "bridge_lib/bridge_state.h"

namespace ble = bridge_learning_env;

enum RolloutResult {
  kWinLose,
  kNumFutureTricks,
  kNumTotalTricks
};

// Wrap dds functions, thus we can call them in multi-threaded context.
class DDSEvaluator {
  public:
    DDSEvaluator() = default;

    static deal PlayStateToDDSdeal(const ble::BridgeState& state);

    static ddTableDeal AuctionStateToDDSddTableDeal(
        const ble::BridgeState& state);

    // Apply move at state and use dds result.
    int Rollout(const ble::BridgeState& state,
                const ble::BridgeMove& move,
                ble::Player result_for,
                RolloutResult rollout_result = kWinLose);

    // Evaluate a state for a player
    int Evaluate(const ble::BridgeState& state, ble::Player result_for,
                 RolloutResult rollout_result);

    // Get all optimal moves by dds.
    std::vector<ble::BridgeMove> DDSMoves(const ble::BridgeState& state);

  private:
    std::mutex m_;
    std::condition_variable cv_;
    bool free_ = true;
};



#endif //DDS_EVALUATOR_H
