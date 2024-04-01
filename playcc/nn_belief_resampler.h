//
// Created by qzz on 2024/1/16.
//

#ifndef BRIDGE_LEARNING_PLAYCC_NN_BELIEF_RESAMPLER_H_
#define BRIDGE_LEARNING_PLAYCC_NN_BELIEF_RESAMPLER_H_

#include "torch_actor.h"

#include "bridge_lib/canonical_encoder.h"
#include "resampler.h"

class NNBeliefResampler : public Resampler {
 public:
  NNBeliefResampler(const std::shared_ptr<TorchActor>& torch_actor,
                    const std::shared_ptr<ble::BridgeGame>& game,
                    const int seed)
      : torch_actor_(torch_actor), encoder_(game), rng_(seed), state_(game) {}

  ~NNBeliefResampler() override = default;

  ResampleResult Resample(const ble::BridgeState& state) override;

  void ResetWithParams(
      const std::unordered_map<std::string, std::string>&) override;

 private:
  rela::TensorDict MakeTensorDictObs(const ble::BridgeState& state) const;

  std::array<int, ble::kNumCards> SampleFromBelief(
      const rela::TensorDict& belief, const ble::BridgeState& state) const;

  std::shared_ptr<TorchActor> torch_actor_;
  const ble::CanonicalEncoder encoder_;
  std::mt19937 rng_;

  // Cache state and belief for multiple time calls.
  ble::BridgeState state_;
  rela::TensorDict belief_;
};

#endif  //BRIDGE_LEARNING_PLAYCC_NN_BELIEF_RESAMPLER_H_
