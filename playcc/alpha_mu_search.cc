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
  // std::cout << "fut.score[0]: " << fut.score[0] << std::endl;
  // std::cout << "is max node: " << IsMaxNode(state) << std::endl;

  if (const bool is_max_node = IsMaxNode(state); is_max_node) {
    // Declarer side wins if future tricks + tricks win >= contract.level + 6
    return fut.score[0] + state.NumDeclarerTricks() >= (contract.level + 6);
  }

  return state.NumDeclarerTricks() + ble::kNumTricks - fut.score[0] >= (contract.level + 6);
}

bool StopSearch(const ble::BridgeState& state,
                const int num_max_moves,
                const std::vector<ble::BridgeState>& worlds,
                const std::vector<bool>& possible_worlds,
                ParetoFront& result) {
  const auto contract = state.GetContract();
  if (state.NumDeclarerTricks() >= contract.level + 6) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 1);
    return true;
  }

  if (const int defense_tricks = state.NumTricksPlayed() - state.NumDeclarerTricks();
      defense_tricks > 13 - (contract.level + 6)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 0);
    return true;
  }

  if (num_max_moves == 0) {
    result = ParetoFront{};
    std::vector<int> game_status(worlds.size());
    for (size_t i = 0; i < worlds.size(); ++i) {
      if (possible_worlds[i]) {
        const bool double_dummy_evaluation = DoubleDummyEvaluation(worlds[i]);
        game_status[i] = double_dummy_evaluation;
      }
    }
    result.Insert({{game_status}, possible_worlds});
    return true;
  }
  return false;
}

std::vector<ble::BridgeMove> GetAllLegalMovesFromPossibleWorlds(const std::vector<ble::BridgeState>& worlds,
                                                                const std::vector<bool>& possible_worlds) {
  if (worlds.empty()) {
    return {};
  }
  const auto game = worlds[0].ParentGame();
  std::vector<ble::BridgeMove> result;
  std::set<int> all_legal_uids;
  for (size_t i = 0; i < worlds.size(); ++i) {
    if (possible_worlds[i]) {
      const auto legal_moves = worlds[i].LegalMoves();
      const auto legal_move_uids = MovesToUids(legal_moves, *game);
      all_legal_uids.insert(legal_move_uids.begin(), legal_move_uids.end());
    }
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

std::vector<ble::BridgeState> GetNextWorlds(const std::vector<ble::BridgeState>& worlds,
                                            const std::vector<bool>& next_possible_worlds,
                                            const ble::BridgeMove& move) {
  std::vector<ble::BridgeState> next_worlds;
  for (size_t i = 0; i < worlds.size(); ++i) {
    if (next_possible_worlds[i]) {
      auto world = worlds[i].Child(move);
      next_worlds.push_back(world);
    }
    else {
      next_worlds.push_back(worlds[i].Clone());
    }
  }
  return next_worlds;
}

// TODO: Find better methods to deal with possible_worlds.
ParetoFront VanillaAlphaMu(const ble::BridgeState& state,
                           const int num_max_moves,
                           const std::vector<ble::BridgeState>& worlds,
                           const std::vector<bool>& possible_worlds) {
  ParetoFront result{};
  const bool stop = StopSearch(state, num_max_moves, worlds, possible_worlds, result);
  std::cout << std::boolalpha << "stop: " << stop << "\n";
  std::cout << "result: \n" << result.ToString() << std::endl;
  if (stop) {
    return result;
  }

  ParetoFront front{};
  if (const bool is_max_node = IsMaxNode(state); !is_max_node) {
    const std::vector<ble::BridgeMove> all_moves = GetAllLegalMovesFromPossibleWorlds(worlds, possible_worlds);
    for (const auto& move : all_moves) {
      auto s = state.Child(move);
      auto next_possible_worlds = GetPossibleWorlds(worlds, move);
      const auto next_worlds = GetNextWorlds(worlds, next_possible_worlds, move);
      std::cout << "Step Min node alpha_mu\n";
      ParetoFront f = VanillaAlphaMu(s, num_max_moves, next_worlds, next_possible_worlds);
      front = ParetoFrontMin(front, f);
    }
  }
  else {
    const std::vector<ble::BridgeMove> all_moves = GetAllLegalMovesFromPossibleWorlds(worlds, possible_worlds);
    for (const auto& move : all_moves) {
      auto s = state.Child(move);
      auto next_possible_worlds = GetPossibleWorlds(worlds, move);
      const auto next_worlds = GetNextWorlds(worlds, next_possible_worlds, move);
      std::cout << "Step Max node alphamu\n";
      ParetoFront f = VanillaAlphaMu(s, num_max_moves - 1, next_worlds, next_possible_worlds);
      front = ParetoFrontMax(front, f);
      std::cout << "front after max opertaion:\n" << front.ToString() << std::endl;
    }
  }
  return front;
}
