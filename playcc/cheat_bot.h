//
// Created by qzz on 2023/10/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_CHEAT_BOT_H_
#define BRIDGE_LEARNING_PLAYCC_CHEAT_BOT_H_
#include "../bridge_lib/bridge_state.h"
#include "play_bot.h"
#include "utils.h"

namespace ble = bridge_learning_env;
class CheatBot : public PlayBot {

 public:
  CheatBot() {
    SetMaxThreads(0);
  }

  ble::BridgeMove Act(const ble::BridgeState &state) override {
    auto legal_moves = state.LegalMoves();
    int num_legal_moves = static_cast<int>(legal_moves.size());
    if (num_legal_moves == 1) {
      return legal_moves[0];
    }
    auto dl = StateToDDSDeal(state);
    futureTricks fut{};

    const int res = SolveBoard(
        dl,
        /*target=*/-1,
        /*solutions=*/1, // We only want one card.
        /*mode=*/2,
        &fut,
        /*threadIndex=*/0);
    if (res != RETURN_NO_FAULT) {
      char error_message[80];
      ErrorMessage(res, error_message);
      std::cerr << "double dummy solver: " << error_message << std::endl;
      std::exit(1);
    }
//    std::cout << "suit: " << fut.suit[0] << std::endl;
//    std::cout << "rank: " << fut.rank[0] << std::endl;
    ble::BridgeMove move{
        /*move_type=*/ble::BridgeMove::Type::kPlay,
        /*suit=*/ble::DDSSuitToSuit(fut.suit[0]),
        /*rank=*/ble::DDSRankToRank(fut.rank[0]),
        /*denomination=*/ble::kInvalidDenomination,
        /*level=*/-1,
        /*other_call=*/ble::kNotOtherCall};
    return move;

  }

};
#endif //BRIDGE_LEARNING_PLAYCC_CHEAT_BOT_H_
