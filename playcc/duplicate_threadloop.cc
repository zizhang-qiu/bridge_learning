//
// Created by qzz on 2023/12/29.
//
#include "duplicate_threadloop.h"

#include "absl/strings/str_format.h"

namespace ble = bridge_learning_env;
void DuplicateThreadloop::mainLoop() {
  while (num_deals_played < cfg_.num_deals) {
    // Generate random deals.
    auto random_deal = ble::Permutation(ble::kNumCards);
    const ble::Player declarer = cfg_.contract.declarer;
    for (ble::Player player = ble::kNorth; player < declarer; ++player) {
      random_deal.push_back(ble::kPass + ble::kBiddingActionBase);
    }
    const ble::BridgeMove bid_move{/*move_type=*/ble::BridgeMove::Type::kAuction,
                                   /*suit=*/ble::kInvalidSuit,
                                   /*rank=*/-1,
                                   /*denomination=*/cfg_.contract.denomination,
                                   /*level=*/cfg_.contract.level,
                                   /*other_call=*/ble::kNotOtherCall};
    random_deal.push_back(ble::default_game->GetMoveUid(bid_move));
    // Three passes.
    for (int i = 0; i < 3; ++i) {
      random_deal.push_back(ble::kPass + ble::kBiddingActionBase);
    }
    auto state1 = ConstructStateFromTrajectory(random_deal, ble::default_game);
    auto state2 = state1.Clone();

    // Duplicate play.
    RunOnce(state1, player1_, defender_);
    RunOnce(state2, player2_, defender_);

    const bool player1_win = state1.Scores()[cfg_.contract.declarer] > 0;
    const bool player2_win = state2.Scores()[cfg_.contract.declarer] > 0;
    if (verbose_) {
      std::cout << absl::StrFormat(
          "Deal %d\nstate1:\n%s\nstate2\n%s\n", num_deals_played, state1.ToString(), state2.ToString());
    }
    EvaluationResult result = EvaluationResult::kSameResults;
    if (player1_win != player2_win) {
      result = player1_win ? kPlayer1Win : kPlayer2Win;
    }
    queue_->Push(result);
    ++num_deals_played;
  }
}
void DuplicateThreadloop::RunOnce(ble::BridgeState& state,
                                  std::shared_ptr<PlayBot>& declarer,
                                  std::shared_ptr<PlayBot>& defender) const {
  resampler_->ResetWithParams({{"seed", std::to_string(num_deals_played)}});
  while (!state.IsTerminal()) {
    ble::BridgeMove move{};
    if (IsActingPlayerDeclarerSide(state)) {
      move = declarer->Act(state);
    }
    else {
      move = defender->Act(state);
    }
    state.ApplyMove(move);
  }
}
