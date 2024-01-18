//
// Created by qzz on 2024/1/16.
//

#ifndef BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_RESAMPLER_H_
#define BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_RESAMPLER_H_

#include "torch_actor.h"

#include "bridge_lib/canonical_encoder.h"
#include "resampler.h"

class TorchActorResampler final : public Resampler {
  public:
    TorchActorResampler(const std::shared_ptr<TorchActor>& torch_actor,
                        const std::shared_ptr<ble::BridgeGame>& game,
                        const int seed) : torch_actor_(torch_actor), encoder_(game), rng_(seed) {
    }

    ~TorchActorResampler() override = default;

    ResampleResult Resample(const ble::BridgeState& state) override;

  private:
    rela::TensorDict MakeTensorDictObs(const ble::BridgeState& state) const;

    std::array<int, ble::kNumCards> SampleFromBelief(const rela::TensorDict& belief,
                                                     const ble::BridgeState& state,
                                                     const rela::TensorDict& obs) const;

    std::shared_ptr<TorchActor> torch_actor_;
    const ble::CanonicalEncoder encoder_;
    std::mt19937 rng_;
};

#endif //BRIDGE_LEARNING_PLAYCC_TORCH_ACTOR_RESAMPLER_H_
