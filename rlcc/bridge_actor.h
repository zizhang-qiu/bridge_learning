//
// Created by qzz on 2024/1/23.
//

#ifndef BRIDGE_ACTOR_H
#define BRIDGE_ACTOR_H
#include "rela/tensor_dict.h"

class BridgeActor {
  public:
    virtual ~BridgeActor() = default;

    virtual rela::TensorDict Act(const rela::TensorDict& obs) =0;

    virtual void SendExperience(const rela::TensorDict& d) {
      std::cerr << "SendExperience function not implemented!" << std::endl;
      std::abort();
    }
};
#endif //BRIDGE_ACTOR_H
