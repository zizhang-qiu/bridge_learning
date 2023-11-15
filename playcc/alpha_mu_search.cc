//
// Created by qzz on 2023/11/14.
//

#include "alpha_mu_search.h"

bool IsMaxNode(const ble::BridgeState& state) {
  const ble::Contract contract = state.GetContract();
  const ble::Player current_player = state.CurrentPlayer();
  return ble::Partnership(contract.declarer) == ble::Partnership(current_player);
}

ble::BridgeState ApplyMove(const ble::BridgeMove& move, ble::BridgeState state) {
  state.ApplyMove(move);
  return state;
}

bool DoubleDummyEvaluation(const ble::BridgeState& state) {
  SetMaxThreads(0);
  const auto dl = StateToDeal(state);
  futureTricks fut{};
  const ble::Contract contract = state.GetContract();
  const int res = SolveBoard(dl,
                             /*target=*/-1,
                             /*solutions=*/1,
                             /*mode=*/2,
                             &fut,
                             /*threadIndex=*/0);
  if (res != RETURN_NO_FAULT) {
    char error_message[80];
    ErrorMessage(res, error_message);
    std::cerr << "double dummy solver: " << error_message << std::endl;
    std::exit(1);
  }
  std::cout << "fut.score[0]: " << fut.score[0] << std::endl;
  std::cout << "is max node: " << IsMaxNode(state) << std::endl;

  if (const bool is_max_node = IsMaxNode(state); is_max_node) {
    // Declarer side wins if future tricks + tricks win >= contract.level + 6
    return fut.score[0] + state.NumDeclarerTricks() >= (contract.level + 6);
  }

  return state.NumDeclarerTricks() + ble::kNumTricks - fut.score[0] >= (contract.level + 6);
}

bool StopSearch(const ble::BridgeState& state,
                const int num_max_moves,
                const std::vector<ble::BridgeState>& worlds,
                ParetoFront& result) {
  const auto contract = state.GetContract();
  if (state.NumDeclarerTricks() >= (contract.level + 6)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(static_cast<int>(worlds.size()), 1);
    return true;
  }

  if (const int defense_tricks = state.NumTricksPlayed() - state.NumDeclarerTricks();
      defense_tricks > 13 - (contract.level + 6)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(static_cast<int>(worlds.size()), 0);
    return true;
  }

  if (num_max_moves == 0) {
    result = ParetoFront{};
    std::vector<int> game_status(worlds.size());
    for (size_t i = 0; i < worlds.size(); ++i) {
      const bool double_dummy_evaluation = DoubleDummyEvaluation(worlds[i]);
      game_status[i] = double_dummy_evaluation;
    }
    result.Insert({{game_status}, std::vector<bool>(worlds.size(), true)});
    return true;
  }
  return false;
}

std::vector<ble::BridgeMove> GetAllLegalMovesFromPossibleWorlds(const std::vector<ble::BridgeState>& worlds) {
  if (worlds.empty()) {
    return {};
  }
  const auto game = worlds[0].ParentGame();
  std::vector<ble::BridgeMove> result;
  std::set<int> all_legal_uids;
  for (const auto& w : worlds) {
    const auto legal_moves = w.LegalMoves();
    const auto legal_move_uids = MovesToUids(legal_moves, *game);
    all_legal_uids.insert(legal_move_uids.begin(), legal_move_uids.end());
  }
  result.reserve(all_legal_uids.size());
  for (const auto uid : all_legal_uids) {
    result.push_back(game->GetMove(uid));
  }
  return result;
}

std::vector<bool> GetPossibleWorlds(const std::vector<ble::BridgeState>& worlds, const ble::BridgeMove& move) {
  std::vector<bool> possible_worlds(worlds.size());
  for (size_t i = 0; i < worlds.size(); ++i) {
    possible_worlds[i] = worlds[i].MoveIsLegal(move);
  }
  return possible_worlds;
}

ParetoFront VanillaAlphaMu(const ble::BridgeState& state,
                           const int num_max_moves,
                           const std::vector<ble::BridgeState>& worlds) {
  if (ParetoFront result{}; StopSearch(state, num_max_moves, worlds, result)) {
    return result;
  }
  ParetoFront front{};
  if (const bool is_max_node = IsMaxNode(state); !is_max_node) {
    const std::vector<ble::BridgeMove> all_moves = GetAllLegalMovesFromPossibleWorlds(worlds);
    for (const auto& move : all_moves) {
      auto s = state.Child(move);
      auto possible_worlds = GetPossibleWorlds(worlds, move);
      ParetoFront f = VanillaAlphaMu(*s, num_max_moves, worlds);
      front = ParetoFrontJoin(front, f);
    }
  }
  else {
    const std::vector<ble::BridgeMove> all_moves = GetAllLegalMovesFromPossibleWorlds(worlds);
    for (const auto& move : all_moves) {
      auto s = state.Child(move);
      auto possible_worlds = GetPossibleWorlds(worlds, move);
      ParetoFront f = VanillaAlphaMu(*s, num_max_moves - 1, worlds);
      front = ParetoFrontJoin(front, f);
    }
  }
  return front;
}
