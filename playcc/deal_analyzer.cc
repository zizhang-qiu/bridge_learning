//
// Created by qzz on 2023/12/22.
//

#include "deal_analyzer.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

std::vector<ble::BridgeMove> DealAnalyzer::DDSMoves(const ble::BridgeState &state) {
  SetMaxThreads(0);
  auto dl = StateToDDSDeal(state);
  futureTricks fut{};

  const int res = SolveBoard(
      dl,
      /*target=*/-1,
      /*solutions=*/2, // We want all the cards.
      /*mode=*/2,
      &fut,
      /*threadIndex=*/0);

  if (res != RETURN_NO_FAULT) {
    char error_message[80];
    ErrorMessage(res, error_message);
    std::cerr << "double dummy solver: " << error_message << std::endl;
    std::exit(1);
  }

  std::vector<ble::BridgeMove> dds_moves = GetMovesFromFutureTricks(fut);
//  for (int i = 0; i < ble::kNumCardsPerHand; ++i) {
//    if (fut.rank[i] != 0) {
//      dds_moves.emplace_back(
//          /*move_type=*/ble::BridgeMove::Type::kPlay,
//          /*suit=*/ble::DDSSuitToSuit(fut.suit[i]),
//          /*rank=*/ble::DDSRankToRank(fut.rank[i]),
//          /*denomination=*/ble::kInvalidDenomination,
//          /*level=*/-1,
//          /*other_call=*/ble::kNotOtherCall);
//    }
//  }
//
//  for (int i = 0; i < ble::kNumCardsPerHand; ++i) {
//    std::cout << absl::StrFormat("equals[%d]=%d", i, fut.equals[i]) << std::endl;
//  }

  return dds_moves;

}
void DealAnalyzer::Analyze(ble::BridgeState state, AlphaMuBot alpha_mu_bot, PIMCBot pimc_bot) {
  while (!state.IsTerminal()) {
    if (IsActingPlayerDeclarerSide(state)) {
      const auto dds_moves = DDSMoves(state);
      const auto move = alpha_mu_bot.Step(state);
      if (std::find(dds_moves.begin(), dds_moves.end(), move) == dds_moves.end() &&
          state.NumDeclarerTricks() < state.GetContract().level + 6) {
        file::File tt_file{absl::StrCat(save_dir_, "/tt.txt"), "w"};
        tt_file.Write(alpha_mu_bot.GetTT().Serialize());
        std::cout << "trajectory:" << VectorToString(state.UidHistory()) << std::endl;
        file::File traj_file{absl::StrCat(save_dir_, "/traj.txt"), "w"};
        traj_file.Write(VectorToString(state.UidHistory()));
        std::cout << absl::StrFormat("At state:\n%s, the alpha bot gives move: %s\nThe dds give moves:",
                                     state.ToString(), move.ToString());
        for(const auto &m:dds_moves){
          std::cout << m << std::endl;
        }
        return;
      }
      state.ApplyMove(move);
    } else {
      const auto move = pimc_bot.Step(state);
      state.ApplyMove(move);
    }
  }
  std::cout << state << std::endl;
}
