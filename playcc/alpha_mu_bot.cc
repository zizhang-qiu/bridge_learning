//
// Created by qzz on 2023/12/7.
//

#include "alpha_mu_bot.h"

#include <algorithm>
#include <vector>

#include "absl/strings/str_cat.h"

#include "bridge_lib/bridge_scoring.h"
#include "bridge_lib/bridge_utils.h"
#include "dll.h"
#include "playcc/common_utils/log_utils.h"
#include "playcc/dds_evaluator.h"
#include "playcc/pareto_front.h"

bool IsFirstMaxNode(const ble::BridgeStateWithoutHiddenInfo& state) {
  return state.NumCardsPlayed() == 1;
}

bool IsFirstMaxNode(const ble::BridgeState& state) {
  return state.NumCardsPlayed() == 1;
}

int Result(const RolloutResult rollout_result, bool is_result_player_declarer,
           int target_tricks, int num_declarer_future_tricks,
           int num_declarer_total_tricks, int num_tricks_left) {
  switch (rollout_result) {
    case kWinLose:
      if (is_result_player_declarer) {
        return num_declarer_total_tricks >= target_tricks;
      }
      return num_declarer_total_tricks < target_tricks;
    case kNumFutureTricks:
      if (is_result_player_declarer) {
        return num_declarer_future_tricks;
      }
      return num_tricks_left - num_declarer_future_tricks;
    case kNumTotalTricks:
      if (is_result_player_declarer) {
        return num_declarer_total_tricks;
      }
      return ble::kNumTricks - num_declarer_total_tricks;
    default:
      SpielFatalError(absl::StrCat("Wrong rollout result: ", rollout_result));
  }
}

std::vector<int> EvaluateWorldsParallel(const Worlds& worlds,
                                        const RolloutResult rollout_result) {
  SetMaxThreads(0);
  const auto& possible = worlds.Possible();
  const auto states = worlds.States();
  ::boards bo{};
  int num_possible_states = 0;
  for (int i = 0; i < states.size(); ++i) {
    if (possible[i]) {
      bo.deals[num_possible_states] =
          DDSEvaluator::PlayStateToDDSdeal(states[i]);
      bo.mode[num_possible_states] = 2;
      bo.solutions[num_possible_states] = 1;
      bo.target[num_possible_states] = -1;
      ++num_possible_states;
    }
  }
  bo.noOfBoards = num_possible_states;
  solvedBoards solved;
  const int return_code = SolveAllBoardsBin(&bo, &solved);
  if (return_code != RETURN_NO_FAULT) {
    char line[80];
    ErrorMessage(return_code, line);
    SpielFatalError(absl::StrCat("double dummy solver error", line));
  }

  // For each possible state, evaluate the result
  int idx = 0;
  std::vector<int> evaluation(worlds.Size());
  for (int i = 0; i < states.size(); ++i) {
    if (possible[i]) {
      const auto& state = states[i];
      const ble::Contract contract = state.GetContract();
      const int target_tricks = contract.level + 6;
      const ble::Player declarer = contract.declarer;
      const ble::Player current_player = state.CurrentPlayer();
      const auto result_player = current_player;
      const bool is_cur_player_declarer =
          ble::Partnership(current_player) == ble::Partnership(declarer);
      const bool is_result_player_declarer = is_cur_player_declarer;
      const auto fut = solved.solvedBoard[idx];
      const int num_tricks_left = ble::kNumTricks - state.NumTricksPlayed();
      const int num_declarer_future_tricks =
          is_cur_player_declarer ? fut.score[0]
                                 : num_tricks_left - fut.score[0];

      const int num_declarer_total_tricks =
          num_declarer_future_tricks + state.NumDeclarerTricks();

      evaluation[i] = Result(rollout_result, true,
                             target_tricks, num_declarer_future_tricks,
                             num_declarer_total_tricks, num_tricks_left);
      ++idx;
    } else {
      evaluation[i] =-1;
    }
  }
  SPIEL_CHECK_EQ(idx, solved.noOfBoards);
  return evaluation;
}

ble::BridgeMove AlphaMuBot::Step(const ble::BridgeState& state) {
  SPIEL_CHECK_FALSE(state.IsTerminal());
  SPIEL_CHECK_EQ(ble::Partnership(state.GetContract().declarer),
                 ble::Partnership(player_id_));
  //  std::cout << "tt size: " << tt_.Table().size() << std::endl;
  if (IsFirstMaxNode(state)) {
    tt_.Clear();
    last_iteration_front_ = std::optional<ParetoFront>{};
  }
  //  const auto &legal_moves = state.LegalMoves();
  const auto legal_moves = GetLegalMovesWithoutEquivalentCards(state);
  const int num_legal_moves = static_cast<int>(legal_moves.size());
  // Only one legal move, return it.
  if (num_legal_moves == 1 && !cfg_.search_with_one_legal_move) {
    return legal_moves[0];
  }

  const auto deals = ResampleMultipleDeals(resampler_, state, cfg_.num_worlds);
  const ParetoFront front =
      Search(ble::BridgeStateWithoutHiddenInfo(state), cfg_.num_max_moves,
             Worlds(deals, state), {});
  last_iteration_front_ = front;
  auto best = front.BestOutcome();
  if (best.move.MoveType() == bridge_learning_env::BridgeMove::kInvalid) {
    best.move = state.LegalMoves()[0];
  }
  //  std::cout << front << std::endl;
  return best.move;
}

std::pair<bool, ParetoFront> AlphaMuBot::Stop(
    const ble::BridgeStateWithoutHiddenInfo& state, int num_max_moves,
    const Worlds& worlds) {

  // Stop when the game is terminated.
  if (state.IsTerminal()) {
    const int num_declarer_tricks = state.NumDeclarerTricks();
    const int target_tricks = state.GetContract().level + 6;
    ParetoFront result{};
    std::vector<int> game_outcomes(worlds.Size(), 0);
    auto possible = worlds.Possible();
    switch (cfg_.rollout_result) {

      case kWinLose: {
        const int win = num_declarer_tricks >= target_tricks;
        std::fill(game_outcomes.begin(), game_outcomes.end(), win);
        result.Insert({game_outcomes, possible});
        return {true, result};
      }
      case kNumFutureTricks:
        result.Insert({game_outcomes, possible});
        return {true, result};
      case kNumTotalTricks:
        std::fill(game_outcomes.begin(), game_outcomes.end(),
                  num_declarer_tricks);
        result.Insert({game_outcomes, possible});
        return {true, result};
      default:
        SpielFatalError(
            absl::StrCat("Wrong rollout result: ", cfg_.rollout_result));
    }
  }

  if (cfg_.rollout_result == kWinLose) {
    const ble::Contract contract = state.GetContract();
    const int target = contract.level + 6;
    if (state.NumDeclarerTricks() >= target) {
      ParetoFront result{};
      std::vector<int> outcomes(worlds.Size(), 1);
      result.Insert({outcomes, worlds.Possible()});
      return {true, result};
    }

    const int num_defender_tricks =
        state.NumTricksPlayed() - state.NumDeclarerTricks();
    if (num_defender_tricks > (ble::kNumTricks - target)) {
      ParetoFront result{};
      std::vector<int> outcomes(worlds.Size(), 0);
      result.Insert({outcomes, worlds.Possible()});
      return {true, result};
    }
  }

  if (num_max_moves == 0) {
    std::vector<int> evaluation(worlds.Size(), 0);
    const auto& states = worlds.States();
    const auto possible = worlds.Possible();
    const ble::Player declarer = state.GetContract().declarer;
    for (size_t i = 0; i < states.size(); ++i) {
      if (possible[i]) {
        evaluation[i] =
            dds_evaluator_.Evaluate(states[i], declarer, cfg_.rollout_result);
      } else {
        evaluation[i] = -1;
      }
    }
    // evaluation = EvaluateWorldsParallel(worlds, cfg_.rollout_result);
    ParetoFront result{};
    result.Insert({evaluation, possible});
    return {true, result};
  }
  return {false, {}};
}

ParetoFront AlphaMuBot::Search(const ble::BridgeStateWithoutHiddenInfo& state,
                               int num_max_moves, const Worlds& worlds,
                               const ParetoFront& alpha) {
  auto [stop, result] = Stop(state, num_max_moves, worlds);
  if (stop) {
    tt_[state] = result;
    return result;
  }

  bool is_state_in_tt = tt_.HasKey(state);

  if (ble::Partnership(state.CurrentPlayer()) !=
      ble::Partnership(state.GetContract().declarer)) {
    // Min node.
    ParetoFront mini{};
    // Early cut.

    if (cfg_.early_cut) {
      if (is_state_in_tt && ParetoFrontDominate(alpha, tt_[state])) {
        //        std::cout << "Perform early cut." << std::endl;
        return mini;
      }
    }
    std::vector<ble::BridgeMove> all_moves = worlds.GetAllPossibleMoves();
    if (is_state_in_tt) {
      auto it = std::find(all_moves.begin(), all_moves.end(),
                          tt_[state].BestOutcome().move);
      if (it != all_moves.end()) {
        std::rotate(all_moves.begin(), it, it + 1);
      }
    }

    for (const auto& move : all_moves) {
      const auto s = state.Child(move);
      const auto next_worlds = worlds.Child(move);
      ParetoFront f = Search(s, num_max_moves, next_worlds, {});
      f.SetMove(move);
      mini = ParetoFrontMin(mini, f);
    }

    tt_[state] = mini;
    return mini;
  } else {
    // Max node.
    ParetoFront front{};
    std::vector<ble::BridgeMove> all_moves = worlds.GetAllPossibleMoves();
    bool tt_move_first = false;
    if (is_state_in_tt) {
      auto it = std::find(all_moves.begin(), all_moves.end(),
                          tt_[state].BestOutcome().move);
      if (it != all_moves.end()) {
        tt_move_first = true;
        std::rotate(all_moves.begin(), it, it + 1);
      }
    }
    for (const auto& move : all_moves) {
      const auto s = state.Child(move);
      const auto next_worlds = worlds.Child(move);
      ParetoFront f = Search(s, num_max_moves - 1, next_worlds, front);
      f.SetMove(move);
      //      if(num_max_moves == cfg_.num_max_moves){
      //        std::cout << "move: " << move.ToString() << ", f:\n" << f << std::endl;
      //      }

      front = ParetoFrontMax(front, f);

      if (cfg_.root_cut) {
        if (num_max_moves == cfg_.num_max_moves) {
          // Root node.
          if (tt_move_first && last_iteration_front_.has_value() &&
              last_iteration_front_->BestOutcome().Score() ==
                  front.BestOutcome().Score()) {
            //  std::cout << "Perform root cut." << std::endl;
            //  std::cout << "Current state:\n" << state << std::endl;
            //  std::cout << "Last iteration front :\n" << last_iteration_front_.value() << std::endl;
            //  std::cout << "front: \n" << front << std::endl;
            break;
          }
        }
      }
    }
    tt_[state] = front;
    return front;
  }
}

std::unique_ptr<PlayBot> MakeAlphaMuBot(ble::Player player_id,
                                        AlphaMuConfig cfg) {
  return std::make_unique<AlphaMuBot>(nullptr, cfg, player_id);
}

namespace {
class AlphaMuFactory : public BotFactory {
 public:
  ~AlphaMuFactory() = default;

  std::unique_ptr<PlayBot> Create(
      std::shared_ptr<const ble::BridgeGame> game, ble::Player player,
      const ble::GameParameters& bot_params) override {
    const int num_max_moves =
        ble::ParameterValue<int>(bot_params, "num_max_moves", 1);
    const int num_worlds =
        ble::ParameterValue<int>(bot_params, "num_worlds", 20);
    const bool early_cut =
        ble::ParameterValue<bool>(bot_params, "early_cut", false);
    const bool root_cut =
        ble::ParameterValue<bool>(bot_params, "root_cut", false);
    const bool use_tt = ble::ParameterValue<bool>(bot_params, "use_tt", false);
    const AlphaMuConfig cfg{num_max_moves, num_worlds, false,
                            use_tt,        early_cut,  root_cut};
    return MakeAlphaMuBot(player, cfg);
  }
};

REGISTER_PLAY_BOT("alpha_mu", AlphaMuFactory);
}  // namespace
