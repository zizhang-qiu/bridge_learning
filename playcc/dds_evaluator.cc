//
// Created by qzz on 2024/1/21.
//

#include "dds_evaluator.h"

#include <absl/strings/str_format.h>

#include "absl/strings/str_cat.h"
#include "bridge_lib/bridge_scoring.h"
#include "bridge_lib/bridge_utils.h"
#include "common_utils/log_utils.h"
#include "utils.h"

deal DDSEvaluator::PlayStateToDDSdeal(const ble::BridgeState& state) const {
  SPIEL_CHECK_EQ(static_cast<int>(state.CurrentPhase()),
                 static_cast<int>(ble::Phase::kPlay));
  deal dl{};
  const ble::Contract contract = state.GetContract();
  // Trump.
  dl.trump = ble::DenominationToDDSStrain(contract.denomination);

  // Leader of current trick.
  const ble::Trick current_trick = state.CurrentTrick();
  if (const ble::Player leader = current_trick.Leader();
      leader != ble::kInvalidPlayer) {
    // Current trick has started for several cards.
    dl.first = leader;
  } else {
    // Current trick have't started.
    if (state.IsDummyActing()) {
      dl.first = state.GetDummy();
    } else {
      dl.first = state.CurrentPlayer();
    }
  }

  // Played trick in current trick.
  const auto play_history = state.PlayHistory();

  const int num_tricks_played = state.NumTricksPlayed();
  const int num_card_played_current_trick =
      state.NumCardsPlayed() - num_tricks_played * ble::kNumPlayers;

  memset(dl.currentTrickSuit, 0, 3 * sizeof(dl.currentTrickSuit));
  memset(dl.currentTrickRank, 0, 3 * sizeof(dl.currentTrickSuit));
  for (int i = 0; i < num_card_played_current_trick; ++i) {
    const ble::BridgeHistoryItem item =
        play_history[num_tricks_played * ble::kNumPlayers + i];
    dl.currentTrickSuit[i] = ble::SuitToDDSSuit(item.suit);
    dl.currentTrickRank[i] = ble::RankToDDSRank(item.rank);
  }

  // Hands of players
  const auto& hands = state.Hands();
  for (const ble::Player pl : ble::kAllSeats) {
    for (const auto card : hands[pl].Cards()) {
      dl.remainCards[pl][SuitToDDSSuit(card.CardSuit())] +=
          1 << ble::RankToDDSRank(card.Rank());
    }
  }

  return dl;
}

ddTableDeal DDSEvaluator::AuctionStateToDDSddTableDeal(
    const ble::BridgeState& state) const {
  SPIEL_CHECK_EQ(static_cast<int>(state.CurrentPhase()),
                 static_cast<int>(ble::Phase::kAuction));
  ddTableDeal dd_table_deal{};
  const auto& hands = state.Hands();
  for (const ble::Player pl : ble::kAllSeats) {
    for (const auto card : hands[pl].Cards()) {
      dd_table_deal.cards[pl][ble::SuitToDDSSuit(card.CardSuit())] +=
          1 << ble::RankToDDSRank(card.Rank());
    }
  }
  return dd_table_deal;
}

int DDSEvaluator::Rollout(const ble::BridgeState& state,
                          const ble::BridgeMove& move,
                          const ble::Player result_for,
                          const RolloutResult rollout_result) {
  //  std::cout << "Enter rollout" << std::endl;
  std::unique_lock<std::mutex> lock(m_);
  cv_.wait(lock, [this] { return free_; });
  free_ = false;
  //  std::cout << "Reach here." << std::endl;
  const bool is_result_player_declarer =
      ble::Partnership(result_for) ==
      ble::Partnership(state.GetContract().declarer);
  const ble::Player declarer = state.GetContract().declarer;
  const auto child = state.Child(move);
  const ble::Player child_player = child.CurrentPlayer();
  const bool is_child_player_declarer =
      ble::Partnership(child_player) == ble::Partnership(declarer);
  const auto dl = PlayStateToDDSdeal(child);
  //  std::cout << "Get dds deal" << std::endl;

  SetMaxThreads(0);

  futureTricks fut{};
  const int res = SolveBoard(dl,
                             /*target=*/-1,    // Maximum number of tricks
                             /*solutions=*/1,  // One card
                             /*mode=*/2,       // Reuse tt
                             &fut,
                             /*threadIndex=*/0);

  if (res != RETURN_NO_FAULT) {
    char error_message[80];
    ErrorMessage(res, error_message);
    SpielFatalError(absl::StrCat("double dummy solver:", error_message));
  }

  lock.unlock();
  cv_.notify_all();
  free_ = true;

  const int num_future_tricks_win_by_child_player = fut.score[0];

  const int num_tricks_left = ble::kNumTricks - child.NumTricksPlayed();
  const int target = state.GetContract().level + 6;

  const int num_declarer_future_tricks =
      is_child_player_declarer
          ? num_future_tricks_win_by_child_player
          : num_tricks_left - num_future_tricks_win_by_child_player;

  if (rollout_result == RolloutResult::kNumFutureTricks) {
    if (is_result_player_declarer) {
      return num_declarer_future_tricks;
    }
    return num_tricks_left - num_declarer_future_tricks;
  }

  if (rollout_result == RolloutResult::kNumTotalTricks) {
    if (is_result_player_declarer) {
      return num_declarer_future_tricks + child.NumDeclarerTricks();
    }
    return num_tricks_left - num_declarer_future_tricks +
           (child.NumTricksPlayed() - child.NumDeclarerTricks());
  }

  if (rollout_result == RolloutResult::kWinLose) {
    if (is_result_player_declarer) {
      // Declarer wins if reach target.
      return static_cast<int>(
          child.NumDeclarerTricks() + num_declarer_future_tricks >= target);
    }
    // Defenders win if target is not reached.
    return static_cast<int>(
        child.NumDeclarerTricks() + num_declarer_future_tricks < target);
  }

  SpielFatalError(absl::StrFormat(
      "Should not reach here, check rollout result: %d.", rollout_result));
}

std::vector<ble::BridgeMove> DDSEvaluator::DDSMoves(
    const ble::BridgeState& state) {
  std::unique_lock<std::mutex> lock(m_);
  cv_.wait(lock, [this] { return free_; });
  free_ = false;
  SPIEL_CHECK_FALSE(state.IsTerminal());
  SPIEL_CHECK_EQ(static_cast<int>(state.CurrentPhase()),
                 static_cast<int>(ble::Phase::kPlay));

  SetMaxThreads(0);
  const auto dl = PlayStateToDDSdeal(state);
  futureTricks fut{};
  const int res = SolveBoard(dl,
                             /*target=*/-1,
                             /*solutions=*/2,  // We want all the cards.
                             /*mode=*/2, &fut,
                             /*threadIndex=*/0);

  if (res != RETURN_NO_FAULT) {
    char error_message[80];
    ErrorMessage(res, error_message);
    SpielFatalError(absl::StrCat("double dummy solver:", error_message));
  }
  lock.unlock();
  cv_.notify_all();
  free_ = true;

  std::vector<ble::BridgeMove> dds_moves = GetMovesFromFutureTricks(fut);
  return dds_moves;
}

int DDSEvaluator::Evaluate(const ble::BridgeState& state,
                           ble::Player result_for,
                           RolloutResult rollout_result) {
  const ble::Contract contract = state.GetContract();
  const ble::Player declarer = contract.declarer;
  const bool is_result_player_declarer =
      ble::Partnership(result_for) == ble::Partnership(declarer);
  const ble::Player current_player = state.CurrentPlayer();
  const bool is_cur_player_declarer =
      ble::Partnership(current_player) == ble::Partnership(declarer);
  const int target_tricks = contract.level + 6;

  // Maybe we can get win/lose result if the declarer side has already won.
  if (rollout_result == RolloutResult::kWinLose) {
    if (state.NumDeclarerTricks() >= target_tricks) {
      return is_result_player_declarer ? 1 : 0;
    }
  }

  // Terminal, no need to call dds.
  if (state.IsTerminal()) {
    const int num_declarer_tricks = state.NumDeclarerTricks();
    switch (rollout_result) {

      case kWinLose:
        return is_cur_player_declarer ? num_declarer_tricks >= target_tricks
                                      : num_declarer_tricks < target_tricks;
      case kNumFutureTricks:
        // Terminal, 0 for both side.
        return 0;
      case kNumTotalTricks:
        return is_result_player_declarer
                   ? num_declarer_tricks
                   : ble::kNumTricks - num_declarer_tricks;
      default:
        SpielFatalError(absl::StrCat("Wrong rollout result: ", rollout_result));
    }
  }

  // Get dds result.
  const auto dl = PlayStateToDDSdeal(state);
  SetMaxThreads(0);
  futureTricks fut{};
  const int res = SolveBoard(dl,
                             /*target=*/-1,    // Maximum number of tricks
                             /*solutions=*/1,  // One card
                             /*mode=*/2,       // Reuse tt
                             &fut,
                             /*threadIndex=*/0);

  if (res != RETURN_NO_FAULT) {
    char error_message[80];
    ErrorMessage(res, error_message);
    SpielFatalError(absl::StrCat("double dummy solver:", error_message));
  }

  // Analyze result.
  const int num_tricks_left = ble::kNumTricks - state.NumTricksPlayed();
  const int num_declarer_future_tricks =
      is_cur_player_declarer ? fut.score[0] : num_tricks_left - fut.score[0];

  const int num_declarer_total_tricks =
      num_declarer_future_tricks + state.NumDeclarerTricks();

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
