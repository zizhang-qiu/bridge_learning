//
// Created by qzz on 2024/1/20.
//

#ifndef TORCH_OPENING_LEAD_BOT_H
#define TORCH_OPENING_LEAD_BOT_H
#include "dds_evaluator.h"
#include "play_bot.h"
#include "nn_belief_resampler.h"

struct BeliefBasedOpeningLeadBotConfig {
  int num_worlds = 20;
  int num_max_sample = 1000;
  // Set to true if we need to sample num_worlds worlds and after num_max_sample
  // times we still haven't got that.
  bool fill_with_uniform_sample = true;
  RolloutResult rollout_result = kWinLose;
  bool verbose = false;
};

class NNBeliefOpeningLeadBot : public PlayBot {
  public:
    NNBeliefOpeningLeadBot(const std::shared_ptr<TorchActor>& torch_actor,
                           const std::shared_ptr<ble::BridgeGame>& game,
                           const int seed,
                           const std::shared_ptr<DDSEvaluator>& evaluator,
                           const BeliefBasedOpeningLeadBotConfig& cfg)
      : resampler_(torch_actor, game, seed),
        uniform_resampler_(seed),
        evaluator_(evaluator),
        cfg_(cfg) {}

    ble::BridgeMove Step(const ble::BridgeState& state) override;

    ~NNBeliefOpeningLeadBot() override = default;

    std::string Name() const override { return "torch opening lead."; }

    bool IsClonable() const override { return false; }

  private:
    NNBeliefResampler resampler_;
    UniformResampler uniform_resampler_;
    std::shared_ptr<DDSEvaluator> evaluator_;
    BeliefBasedOpeningLeadBotConfig cfg_;

};
#endif //TORCH_OPENING_LEAD_BOT_H
