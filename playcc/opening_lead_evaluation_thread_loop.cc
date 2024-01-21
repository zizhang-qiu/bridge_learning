//
// Created by qzz on 2024/1/21.
//
#include "opening_lead_evaluation_thread_loop.h"

#include "deal_analyzer.h"

void OpeningLeadEvaluationThreadLoop::mainLoop() {
  int num_games_evaluated = 0;
  int num_match = 0;
  for (const auto& trajectory : trajectories_) {
    ble::BridgeState state{game_};
    int idx = 0;
    // Deal and auction.
    while (state.CurrentPhase() != ble::Phase::kPlay) {
      ble::BridgeMove move{};
      if (state.IsChanceNode()) {
        move = game_->GetChanceOutcome(trajectory[idx]);
      } else {
        move = game_->GetMove(trajectory[idx]);
      }
      state.ApplyMove(move);
      ++idx;
    }

    // Get DDS moves (optimal).
    const auto dds_moves = dds_evaluator_->DDSMoves(state);
    if (verbose_)
      std::cout << absl::StrFormat("Thread %d get dds moves.", thread_idx_) << std::endl;

    // Get bot's move.
    const auto bot_move = bot_->Step(state);
    if (verbose_)
      std::cout << absl::StrFormat("Thread %d get bot's move.", thread_idx_) << std::endl;

    const bool match = std::find(dds_moves.begin(), dds_moves.end(), bot_move)
                       != dds_moves.end();

    ++num_games_evaluated;
    if (match)
      ++num_match;

    if (verbose_)
      std::cout << absl::StrFormat("Thread %d, %d/%d, total: %d.",
                                   thread_idx_, num_match, num_games_evaluated,
                                   trajectories_.size()) << std::endl;
    bot_evaluation_->Push(match);
  }
}
