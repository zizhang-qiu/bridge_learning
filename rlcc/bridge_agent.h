//
// Created by qzz on 2023/10/10.
//

#ifndef BRIDGE_LEARNING_RLCC_BRIDGE_AGENT_H_
#define BRIDGE_LEARNING_RLCC_BRIDGE_AGENT_H_
#include "rela/batch_runner.h"

class BridgeAgent {
 private:
  std::shared_ptr<rela::BatchRunner> runner_;
};

#endif //BRIDGE_LEARNING_RLCC_BRIDGE_AGENT_H_
