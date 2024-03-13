#ifndef RLCC_GAME_ACTOR_TRANSITION_BUFFER_H
#define RLCC_GAME_ACTOR_TRANSITION_BUFFER_H
#include <ATen/core/TensorBody.h>
#include <vector>
#include "rela/tensor_dict.h"
#include "rela/transition.h"
#include "rela/types.h"

namespace rlcc {
class GameActorTransitionBuffer {
 public:
  GameActorTransitionBuffer(const float gamma) : gamma_(gamma) {}

  void PushObsAndReply(const rela::TensorDict& obs,
                       const rela::TensorDict& reply) {
    reply_history_.push_back(reply);
    obs_history_.push_back(obs);
  }

  void SetTerminalAndReward(const float reward){
    reward_ = reward;
  }

  size_t Size() const{
    return obs_history_.size();
  }

//   rela::Transition PopTransition(){
//     rela::TensorDict d = rela::tensor_dict::stack(obs_history_, 0);
    
//   }

 private:
  float gamma_;
  std::vector<rela::TensorDict> reply_history_;
  std::vector<rela::TensorDict> obs_history_;
  float reward_;
};
}  // namespace rlcc

#endif /* RLCC_GAME_ACTOR_TRANSITION_BUFFER_H */
