//
// Created by qzz on 2023/11/14.
//
#include "utils.h"
#include "bridge_lib/bridge_utils.h"
std::vector<ble::BridgeHistoryItem> GetPlayHistory(const std::vector<ble::BridgeHistoryItem>& history) {
  std::vector<ble::BridgeHistoryItem> play_history;
  for (const auto item : history) {
    if (item.move.MoveType() == ble::BridgeMove::Type::kPlay) {
      play_history.push_back(item);
    }
  }
  return play_history;
}
std::array<int, ble::kNumCards> HandsToCardIndices(const std::vector<ble::BridgeHand>& hands) {
  std::array<int, ble::kNumCards> res{};
  for (int i = 0; i < ble::kNumCardsPerHand; ++i) {
    for (int pl = 0; pl < ble::kNumPlayers; ++pl) {
      res[i * ble::kNumPlayers + pl] = hands[pl].Cards()[i].Index();
    }
  }
  return res;
}
ble::BridgeState ConstructStateFromDeal(const std::array<int, ble::kNumCards> deal,
                                        const std::shared_ptr<ble::BridgeGame>& game) {
  auto state = ble::BridgeState(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state.ApplyMove(move);
  }
  return state;
}
ble::BridgeState ConstructStateFromDeal(const std::array<int, ble::kNumCards>& deal,
                                        const std::shared_ptr<ble::BridgeGame>& game,
                                        const ble::BridgeState& original_state) {
  auto state = ble::BridgeState(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state.ApplyMove(move);
  }
  const auto& history = original_state.History();
  for (int i = ble::kNumCards; i < history.size(); ++i) {
    ble::BridgeMove move = history[i].move;
    state.ApplyMove(move);
  }
  return state;
}
deal StateToDDSDeal(const ble::BridgeState& state) {
  // Should be play phase or game over.
  SPIEL_CHECK_GE(static_cast<int>(state.CurrentPhase()), static_cast<int>(ble::Phase::kPlay));
  deal dl{};
  const ble::Contract contract = state.GetContract();
  dl.trump = ble::DenominationToDDSStrain(contract.denomination);
  const ble::Trick current_trick = state.CurrentTrick();
  dl.first = current_trick.Leader() != ble::kInvalidPlayer ? current_trick.Leader()
      : state.IsDummyActing()                              ? state.GetDummy()
                                                           : state.CurrentPlayer();

  const auto& history = state.History();
  std::vector<ble::BridgeHistoryItem> play_history;
  for (const auto move : history) {
    if (move.move.MoveType() == ble::BridgeMove::Type::kPlay) {
      play_history.push_back(move);
    }
  }

  const int num_tricks_played = static_cast<int>(play_history.size()) / ble::kNumPlayers;
  const int num_card_played_current_trick =
      static_cast<int>(play_history.size()) - num_tricks_played * ble::kNumPlayers;
  memset(dl.currentTrickSuit, 0, 3 * sizeof(dl.currentTrickSuit));
  memset(dl.currentTrickRank, 0, 3 * sizeof(dl.currentTrickSuit));
  for (int i = 0; i < num_card_played_current_trick; ++i) {
    ble::BridgeHistoryItem item = play_history[num_tricks_played * ble::kNumPlayers + i];
    dl.currentTrickSuit[i] = ble::SuitToDDSSuit(item.suit);
    dl.currentTrickRank[i] = ble::RankToDDSRank(item.rank);
  }

  //  std::cout << "currentTrickSuit: ";
  //  for(int i : dl.currentTrickSuit){
  //    std::cout << i << std::endl;
  //  }
  //
  //  std::cout << "currentTrickRank: ";
  //  for(int i : dl.currentTrickRank){
  //    std::cout << i << std::endl;
  //  }

  const auto& hands = state.Hands();
  for (const ble::Player pl : ble::kAllSeats) {
    for (const auto card : hands[pl].Cards()) {
      dl.remainCards[pl][SuitToDDSSuit(card.CardSuit())] += 1 << (2 + card.Rank());
    }
  }

  //  futureTricks fut{};
  //  const int res = SolveBoard(
  //      dl,
  //      /*target=*/-1,
  //      /*solutions=*/1,
  //      /*mode=*/2,
  //      &fut,
  //      /*threadIndex=*/0);
  //  if (res != RETURN_NO_FAULT){
  //    char error_message[80];
  //    ErrorMessage(res, error_message);
  //    std::cerr << "double dummy solver: " << error_message << std::endl;
  //    std::exit(1);
  //  }
  return dl;
}

std::vector<int> MovesToUids(const std::vector<ble::BridgeMove>& moves, const ble::BridgeGame& game) {
  std::vector<int> uids;
  for (const auto& move : moves) {
    const int uid = game.GetMoveUid(move);
    uids.push_back(uid);
  }
  return uids;
}
bool IsActingPlayerDeclarerSide(const ble::BridgeState& state) {
  const auto declarer = state.GetContract().declarer;
  const auto cur_player = state.CurrentPlayer();
  return ble::Partnership(declarer) == ble::Partnership(cur_player);
}

void DefaultErrorHandler(const std::string& error_msg) {
  std::cerr << "Spiel Fatal Error: " << error_msg << std::endl << std::endl << std::flush;
  std::exit(1);
}