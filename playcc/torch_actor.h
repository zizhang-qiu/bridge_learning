//
// Created by qzz on 2024/1/14.
//

#ifndef BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_H_
#define BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_H_

#include "rela/model.h"
#include "rela/model_locker.h"
#include "rela/model_common.h"

#include "bridge_lib/bridge_state.h"
#include "rela/prioritized_replay2.h"

namespace ble = bridge_learning_env;

class TorchActor {
  public:
    TorchActor(const std::shared_ptr<rela::ModelLocker>& model_locker): model_locker_(model_locker) {
    }

    [[nodiscard]] rela::TensorDict GetPolicy(const rela::TensorDict& obs) const;

    [[nodiscard]] rela::TensorDict GetBelief(const rela::TensorDict& obs) const;

  private:
    std::shared_ptr<rela::ModelLocker> model_locker_;
};

#endif //BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_H_
