//
// Created by qzz on 2024/1/9.
//
#include "dds_bot.h"

ble::BridgeMove DDSBot::Step(const ble::BridgeState& state) {
  const auto legal_moves = state.LegalMoves();
  const int num_legal_moves = static_cast<int>(legal_moves.size());
  if (num_legal_moves == 1) {
    return legal_moves[0];
  }
  const auto dl = StateToDDSDeal(state);
  futureTricks fut{};

  const int res = SolveBoard(dl,
                             /*target=*/-1,
                             /*solutions=*/1,
                             // We only want one card.
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
  const ble::BridgeMove move{
      /*move_type=*/ble::BridgeMove::Type::kPlay,
                    /*suit=*/ble::DDSSuitToSuit(fut.suit[0]),
                    /*rank=*/ble::DDSRankToRank(fut.rank[0]),
                    /*denomination=*/ble::kInvalidDenomination,
                    /*level=*/-1,
                    /*other_call=*/ble::kNotOtherCall
  };
  return move;
}

std::unique_ptr<PlayBot> MakeDDSBot(ble::Player player_id) {
  return std::make_unique<DDSBot>();
}

namespace {
class DDSFactory : public BotFactory {
  public:
    ~DDSFactory() = default;

    std::unique_ptr<PlayBot> Create(std::shared_ptr<const ble::BridgeGame> game,
                                    ble::Player player,
                                    const ble::GameParameters&
                                    bot_params) override {
      return MakeDDSBot(player);
    }
};

REGISTER_PLAY_BOT("dds", DDSFactory);
}
