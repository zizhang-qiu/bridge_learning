//
// Created by qzz on 2024/1/14.
//

#ifndef BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_H_
#define BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_H_

#include "rela/batch_runner.h"

#include "bridge_lib/bridge_state.h"
#include "rela/prioritized_replay2.h"

namespace ble = bridge_learning_env;

class TorchActor {
  public:
    TorchActor(const std::shared_ptr<rela::BatchRunner>& runner)
      : runner_(runner) {}

    [[nodiscard]] rela::TensorDict GetPolicy(const rela::TensorDict& obs);

    [[nodiscard]] rela::TensorDict GetBelief(const rela::TensorDict& obs);

  private:
    std::shared_ptr<rela::BatchRunner> runner_;
    rela::FutureReply fut_policy_{};
    rela::FutureReply fut_belief_{};
};

#endif //BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_H_
