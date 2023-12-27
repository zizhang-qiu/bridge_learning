//
// Created by qzz on 2023/11/14.
//
#include <set>
#include "alpha_mu_search.h"
#include "absl/strings/str_cat.h"

bool IsMaxNode(const ble::BridgeState &state) {
  const ble::Contract contract = state.GetContract();
  const ble::Player current_player = state.CurrentPlayer();
  return ble::Partnership(contract.declarer) == ble::Partnership(current_player);
}

bool IsPlayerDeclarerSide(const ble::BridgeState &state, ble::Player player) {
  const ble::Player declarer = state.GetContract().declarer;
  return ble::Partnership(player) == ble::Partnership(declarer);
}

bool IsPlayerDeclarerSide(const ble::BridgeStateWithoutHiddenInfo &state, ble::Player player) {
  const ble::Player declarer = state.GetContract().declarer;
  return ble::Partnership(player) == ble::Partnership(declarer);
}

ble::BridgeState ApplyMove(const ble::BridgeMove &move, ble::BridgeState state) {
  state.ApplyMove(move);
  return state;
}

bool DoubleDummyEvaluation(const ble::BridgeState &state) {
  SetMaxThreads(0);
  const auto dl = StateToDDSDeal(state);
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
    SpielFatalError(absl::StrCat("double dummy solver:", error_message, "current state:\n", state.ToString()));
  }
  //   std::cout << "fut.score[0]: " << fut.score[0] << std::endl;
  //   std::cout << "is max node: " << IsMaxNode(state) << std::endl;
  //   std::cout << "num declarer tricks: " << state.NumDeclarerTricks() << std::endl;

  if (const bool is_max_node = IsMaxNode(state); is_max_node) {
    //    std::cout << "return evaluation at max node" << std::endl;
    //    std::cout << "contract level + 6: " << contract.level + 6 << std::endl;
    // Declarer side wins if future tricks + tricks win >= contract.level + 6
    return fut.score[0] + state.NumDeclarerTricks() >= (contract.level + 6);
  }
  const int num_tricks_left = ble::kNumTricks - state.NumTricksPlayed();

  return state.NumDeclarerTricks() + num_tricks_left - fut.score[0] >= (contract.level + 6);
//  return fut.score[0] + state.NumDeclarerTricks() >= (contract.level + 6);
}

std::vector<int> DoubleDummyEvaluation(const Worlds &worlds) {
  const auto &states = worlds.States();
  const size_t size = states.size();
  const auto possible = worlds.Possible();
  std::vector<int> evaluation(size, 0);
  for (size_t i = 0; i < size; ++i) {
    evaluation[i] = possible[i] ? DoubleDummyEvaluation(states[i]) : 1;
  }
  return evaluation;
}

bool CheckAlreadyWin(const ble::BridgeState &state) {
  const auto contract = state.GetContract();
  return state.NumDeclarerTricks() >= contract.level + 6;
}

bool CheckAlreadyLose(const ble::BridgeState &state) {
  const auto contract = state.GetContract();
  const int defense_tricks = state.NumTricksPlayed() - state.NumDeclarerTricks();
  return defense_tricks > ble::kNumTricks - (contract.level + 6);
}

bool StopSearch(const ble::BridgeState &state,
                const int num_max_moves,
                const std::vector<ble::BridgeState> &worlds,
                const std::vector<int> &possible_worlds,
                ParetoFront &result) {
  if (CheckAlreadyWin(state)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 1);
    return true;
  }

  if (CheckAlreadyLose(state)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 0);
    return true;
  }

  // Leaf node.
  if (num_max_moves == 0) {
    result = ParetoFront{};
    std::vector<int> game_status(worlds.size());
    for (size_t i = 0; i < worlds.size(); ++i) {
      if (possible_worlds[i]) {
        const bool double_dummy_evaluation = DoubleDummyEvaluation(worlds[i]);
        game_status[i] = double_dummy_evaluation;
      }
    }
    result.Insert({game_status, possible_worlds});
    return true;
  }
  return false;
}

std::vector<ble::BridgeMove> GetAllLegalMovesFromPossibleWorlds(const std::vector<ble::BridgeState> &worlds,
                                                                const std::vector<bool> &possible_worlds) {
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

std::vector<bool> GetPossibleWorlds(const ble::BridgeStateWithoutHiddenInfo &state,
                                    const std::vector<ble::BridgeState> &worlds,
                                    const ble::BridgeMove &move) {
  std::vector<bool> possible_worlds(worlds.size());
  for (size_t i = 0; i < worlds.size(); ++i) {
    possible_worlds[i] =
        GetPlayHistory(worlds[i].History()).size() == state.PlayHistory().size() - 1 && worlds[i].MoveIsLegal(move);
  }
  return possible_worlds;
}

bool StopSearch(const ble::BridgeState &state,
                int num_max_moves,
                const Worlds &worlds,
                const vector<int> &possible_worlds,
                ParetoFront &result) {
  if (CheckAlreadyWin(state)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 1);
    return true;
  }
  if (CheckAlreadyLose(state)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 0);
    return true;
  }
  if (num_max_moves == 0) {
    result = ParetoFront{};
    const auto game_status = DoubleDummyEvaluation(worlds);
    result.Insert({game_status, possible_worlds});
    return true;
  }
  return false;
}

std::vector<ble::BridgeState> GetNextWorlds(const std::vector<ble::BridgeState> &worlds,
                                            const std::vector<bool> &next_possible_worlds,
                                            const ble::BridgeMove &move) {
  std::vector<ble::BridgeState> next_worlds;
  for (size_t i = 0; i < worlds.size(); ++i) {
    if (next_possible_worlds[i]) {
      auto world = worlds[i].Child(move);
      next_worlds.push_back(world);
    } else {
      next_worlds.push_back(worlds[i]);
    }
  }
  return next_worlds;
}
bool StopSearch(const ble::BridgeStateWithoutHiddenInfo &state,
                int num_max_moves,
                const vector<ble::BridgeState> &worlds,
                const vector<int> &possible_worlds,
                ParetoFront &result) {
  if (state.NumDeclarerTricks() >= state.GetContract().level + 6) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 1);
    return true;
  }
  const auto contract = state.GetContract();
  const int defense_tricks = state.NumTricksPlayed() - state.NumDeclarerTricks();
  if (defense_tricks >= ble::kNumTricks - (contract.level + 6)) {
    result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 0);
    return true;
  }
  // Leaf node.
  if (num_max_moves == 0) {
    result = ParetoFront{};
    std::vector<int> game_status(worlds.size());
    for (size_t i = 0; i < worlds.size(); ++i) {
      if (possible_worlds[i]) {
        const bool double_dummy_evaluation = DoubleDummyEvaluation(worlds[i]);
        game_status[i] = double_dummy_evaluation;
      } else {
        game_status[i] = 1;
      }
    }
    result.Insert({game_status, possible_worlds});
    return true;
  }
  return false;
}

StopResult StopSearch(const ble::BridgeStateWithoutHiddenInfo &state, int num_max_moves, const Worlds &worlds) {
  const ble::Contract contract = state.GetContract();
  const auto possible_worlds = worlds.Possible();
  const auto last_history = state.PlayHistory().back();

  // Declarer side has already won.
  if (state.NumDeclarerTricks() >= (6 + contract.level)) {
    const ParetoFront result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 1);
    return {true, result};
  }

  // Declarer side has already lost.
  const int defense_tricks = state.NumTricksPlayed() - state.NumDeclarerTricks();
  if (defense_tricks > ble::kNumTricks - (contract.level + 6)) {
    const ParetoFront result = ParetoFront::ParetoFrontWithOneOutcomeVector(possible_worlds, 0);
    return {true, result};
  }

  if (num_max_moves == 0) {
    // Reach leaf node.
    const auto evaluation = DoubleDummyEvaluation(worlds);
    ParetoFront result{};
    result.Insert({evaluation, possible_worlds});
    result.SetMove(last_history.move);
    return {true, result};
  }

  return {false, {}};
}

ParetoFront VanillaAlphaMu(const ble::BridgeStateWithoutHiddenInfo &state, int num_max_moves, const Worlds &worlds) {
  auto [stop, result] = StopSearch(state, num_max_moves, worlds);
  //  std::cout << std::boolalpha << "stop: " << stop << "\n";
  //  std::cout << "result: \n" << result.ToString() << std::endl;
  if (stop) {
    return result;
  }

  ParetoFront front{};
  if (ble::Partnership(state.CurrentPlayer()) != ble::Partnership(state.GetContract().declarer)) {
    // Min node.
    //    std::cout << "Min node, M = " << num_max_moves << std::endl;
    const std::vector<ble::BridgeMove> all_moves = worlds.GetAllPossibleMoves();
    //    std::cout << "Current state: " << state.ToString() << std::endl;
    //    std::cout << "all moves: \n";
    //    for (const auto &move : all_moves) {
    //      std::cout << move.ToString() << "\n";
    //    }
//    std::cout << "f for state \n" << state << std::endl;
    for (const ble::BridgeMove &move : all_moves) {
//      std::cout << "Trying move " << move.ToString() << " in Min node, M = " << num_max_moves << std::endl;
      const auto s = state.Child(move);

      const auto next_worlds = worlds.Child(move);
      ParetoFront f = VanillaAlphaMu(s, num_max_moves, next_worlds);
//      std::cout << "front at Min node, M = " << num_max_moves << ", move: " << move.ToString() << "\n"
//                << f.ToString() << std::endl;
      f.SetMove(move);
      front = ParetoFrontMin(front, f);
      //      std::cout << "Min node, move: " << move.ToString() << "\nfront after join:\n" << front.ToString() <<
      //      std::endl;
    }
//    std::cout << "overall front at Min node, M = " << num_max_moves << "\n" << front.ToString() << std::endl;
  } else {
    // Max node.
    //    std::cout << "Max node, M = " << num_max_moves << std::endl;
    const std::vector<ble::BridgeMove> all_moves = worlds.GetAllPossibleMoves();
    //    std::cout << "Current state: " << state.ToString() << std::endl;
    //    std::cout << "all moves: \n";
    //    for (const auto &move : all_moves) {
    //      std::cout << move.ToString() << "\n";
    //    }
    for (const auto &move : all_moves) {
//      std::cout << "Trying move " << move.ToString() << " in Max node, M = " << num_max_moves << std::endl;

      auto s = state.Child(move);

      const auto next_worlds = worlds.Child(move);
      //      std::cout << "Step Max node alphamu\n";
      ParetoFront f = VanillaAlphaMu(s, num_max_moves - 1, next_worlds);
      f.SetMove(move);
//      if (num_max_moves == 2) {
//        std::cout << "front at Max node, M = " << num_max_moves << ", move: " << move.ToString() << "\n"
//                  << f.ToString() << "\n score: " << f.Score() << std::endl;
//      }
      front = ParetoFrontMax(front, f);
//      if (num_max_moves == 2)
//        std::cout << "Max node, move: " << move.ToString() << "\nfront:\n" << front.ToString() << std::endl;
    }
//    std::cout << "overall front at Max node, M = " << num_max_moves << "\n" << front.ToString() << std::endl;
  }
  return front;
}
