#ifndef RLCC_RNN_BUFFER_H
#define RLCC_RNN_BUFFER_H
#include "rela/tensor_dict.h"
#include "rela/logging.h"
#include "rela/transition.h"

namespace rlcc {
class RNNTransitionBuffer {
 public:
  RNNTransitionBuffer(int max_len, float gamma)
      : max_len_(max_len),
        gamma_(gamma),
        seq_len_(0),
        call_order_(0),
        obs_(max_len),
        action_(max_len),
        terminated_(false),
        reward_(0) {}

  void Init(const rela::TensorDict &h0) {
    h0_ = h0;
  }

  int Len() const { return seq_len_; }

  void PushObs(const rela::TensorDict &obs) {
    RELA_CHECK_EQ(call_order_, 0);
    ++call_order_;

    RELA_CHECK_LT(seq_len_, max_len_);
    obs_[seq_len_] = obs;
  }

  void PushAction(const rela::TensorDict &action) {
    RELA_CHECK_EQ(call_order_, 1);
    action_[seq_len_] = action;
    ++seq_len_;
    call_order_ = 0;
  }

  void PushTerminal() {
    RELA_CHECK_FALSE(terminated_);
    terminated_ = true;
    call_order_ = 0;
  }

  void PushReward(const rela::TensorDict &reward) {
    reward_ = reward.at("r").item<float>();
  }

  rela::RNNTransition PopTransition() {
    RELA_CHECK(terminated_);
    // Cumulative reward.
    std::vector<float> rewards(max_len_);
    rewards[seq_len_ - 1] = reward_;
    for (int i = seq_len_ - 2; i >= 0; --i) {
      rewards[i] += rewards[i + 1] * gamma_;
    }

    // Construct transition.
    torch::Tensor terminal = torch::zeros(max_len_);
    for (size_t i = seq_len_ - 1; i < max_len_; ++i) {
      terminal[i] = 1.0;
    }

    for (size_t i = seq_len_; i < max_len_; ++i) {
      obs_[i] = rela::tensor_dict::zerosLike(obs_[seq_len_ - 1]);
      action_[i] = rela::tensor_dict::zerosLike(action_[seq_len_ - 1]);
    }

    rela::RNNTransition transition;
    transition.h0 = h0_;
    transition.obs = rela::tensor_dict::stack(obs_, 0);
    transition.action = rela::tensor_dict::stack(action_, 0);
    transition.terminal = terminal;
    transition.reward = torch::tensor(rewards);
    transition.seqLen = torch::tensor(seq_len_);

    transition.bootstrap = torch::ones(max_len_);

    seq_len_ = 0;
    terminated_ = false;
    call_order_ = 0;

    return transition;
  }

 private:
  const int max_len_;
  const float gamma_;

  rela::TensorDict h0_;
  std::vector<rela::TensorDict> obs_;
  std::vector<rela::TensorDict> action_;
  bool terminated_;
  float reward_;

  int seq_len_;
  int call_order_;
};
}  // namespace rlcc

#endif /* RLCC_RNN_BUFFER_H */
