//
// Created by qzz on 2024/1/20.
//

#include "torch_opening_lead_bot.h"

#include "pimc.h"
#include "absl/strings/str_cat.h"

ble::BridgeMove TorchOpeningLeadBot::Step(const ble::BridgeState& state) {
  // Check if the state needs a opening lead.
  if (state.CurrentPhase() != ble::Phase::kPlay
      || state.NumCardsPlayed() != 0) {
    SpielFatalError(absl::StrCat(
        "Torch opening lead bot can only play opening lead, but the state is",
        state.ToString()));
  }

  // We don't need to check if there is only one legal move here,
  // because opening lead always have 13 moves.

  int num_sampled_worlds = 0;
  int num_sample_times = 0;

  if (cfg_.verbose) {
    std::cout << "Start sample." << std::endl;
  }
  std::vector<ble::BridgeState> states;
  states.reserve(cfg_.num_worlds);
  // Sample from belief
  while (num_sample_times < cfg_.num_max_sample
         && num_sampled_worlds < cfg_.num_worlds) {
    const auto resample_result = resampler_.Resample(state);
    // std::cout << "result: " << resample_result.success << std::endl;
    ++num_sample_times;
    if (resample_result.success) {
      const auto sampled_state = ConstructStateFromDealAndOriginalState(
          resample_result.result, state.ParentGame(), state);
      states.push_back(sampled_state);
      ++num_sampled_worlds;
    }
  }
  if (cfg_.verbose) {
    std::cout << absl::StrFormat("Finish sample from belief, %d deals sampled",
                                 num_sampled_worlds) << std::endl;
  }

  // Fill remained deals with uniform sampling.
  if (cfg_.fill_with_uniform_sample) {
    while (num_sampled_worlds < cfg_.num_worlds) {
      const auto resample_result = uniform_resampler_.Resample(state);
      // std::cout << "result: " << resample_result.success << std::endl;
      ++num_sample_times;
      if (resample_result.success) {
        const auto sampled_state = ConstructStateFromDealAndOriginalState(
            resample_result.result, state.ParentGame(), state);
        states.push_back(sampled_state);
        ++num_sampled_worlds;
      }
    }
  }

  if (cfg_.verbose) {
    std::cout << absl::StrFormat("Finish uniform sample, %d deals sampled",
                                 num_sampled_worlds) << std::endl;
  }

  SPIEL_CHECK_LE(states.size(), cfg_.num_worlds);

  const auto legal_moves = state.LegalMoves();
  const int num_legal_moves = static_cast<int>(legal_moves.size());
  SearchResult res{};
  res.moves = legal_moves;
  res.scores = std::vector<int>(num_legal_moves, 0);

  // For each sampled worlds, use double dummy result as evaluation.
  for (const auto& sampled_state : states) {
    for (int i = 0; i < num_legal_moves; ++i) {
      const int score = evaluator_->Rollout(sampled_state, legal_moves[i],
                                            state.CurrentPlayer());
      res.scores[i] += score;
    }
  }

  if (cfg_.verbose)
    PrintSearchResult(res);

  auto [move, score] = GetBestAction(res);

  return move;
}
