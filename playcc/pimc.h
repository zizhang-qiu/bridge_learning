//
// Created by qzz on 2023/10/22.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PIMC_H_
#define BRIDGE_LEARNING_PLAYCC_PIMC_H_
#include <utility>

#include "bridge_lib/bridge_state_2.h"
#include "resampler.h"
//#include "rela/logging.h"
#include "bridge_lib/third_party/dds/include/dll.h"
namespace ble = bridge_learning_env;
deal StateToDeal(const ble::BridgeState2 &state) {
  if (state.CurrentPhase() != ble::BridgeState2::Phase::kPlay) {
    std::cerr << "Should be play phase." << std::endl;
    std::abort();
  }
  deal dl{};
  const ble::Contract contract = state.GetContract();
  dl.trump = ble::DenominationToDDSStrain(contract.denomination);
//  std::cout << "dl.trump: " << dl.trump << std::endl;
  const ble::Trick current_trick = state.CurrentTrick();
  dl.first = current_trick.Leader() != ble::kInvalidPlayer ? current_trick.Leader() : state.CurrentPlayer();
//  std::cout << "dl.first: " << dl.first << std::endl;

  const auto &history = state.History();
  std::vector<ble::BridgeHistoryItem> play_history;
  for (const auto move : history) {
    if (move.move.MoveType() == ble::BridgeMove::Type::kPlay) {
      play_history.push_back(move);
    }
  }

  int num_tricks_played = static_cast<int>(play_history.size()) / ble::kNumPlayers;
  int num_card_played_current_trick = static_cast<int>(play_history.size()) - num_tricks_played * ble::kNumPlayers;
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

  const auto &hands = state.Hands();
  for (ble::Player pl : ble::kAllSeats) {
    for (const auto card : hands[pl].Cards()) {
      dl.remainCards[pl][SuitToDDSSuit(card.CardSuit())] += 1
          << (2 + card.Rank());
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

int Rollout(const ble::BridgeState2 &state, ble::BridgeMove move) {
  auto cloned = state.Clone();
  cloned->ApplyMove(move);
  auto dl = StateToDeal(*cloned);
  futureTricks fut{};

  const int res = SolveBoard(
      dl,
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
//  std::cout << fut.nodes << std::endl;
//  std::cout << fut.cards << std::endl;
//  std::cout << fut.rank[0] << std::endl;
//  std::cout << fut.suit[0] << std::endl;
//  std::cout << fut.score[0] << std::endl;
  int num_tricks_left = 13 - state.NumTricksPlayed();
  return num_tricks_left - fut.score[0];
//  return fut.score[0];
}

struct SearchResult {
  std::vector<ble::BridgeMove> moves;
  std::vector<int> scores;
};

std::pair<ble::BridgeMove, int> GetBestAction(const SearchResult &res) {
  auto it = std::max_element(res.scores.begin(), res.scores.end());
  int index = std::distance(res.scores.begin(), it);
  return std::make_pair(res.moves[index], res.scores[index]);
}

class PIMCBot {
 public:
  PIMCBot(std::shared_ptr<Resampler> resampler, int num_sample)
      : resampler_(std::move(resampler)), num_sample_(num_sample) {
    SetMaxThreads(0);
  }

  SearchResult Search(const ble::BridgeState2 &state) {

    auto legal_moves = state.LegalMoves();
    int num_legal_moves = static_cast<int>(legal_moves.size());
//    std::cout << "num legal moves: " << num_legal_moves << std::endl;
    SearchResult res{};
    res.moves = legal_moves;
    res.scores = std::vector<int>(num_legal_moves, 0);

    if (num_legal_moves == 1) {
      return res;
    }
    for (int i = 0; i < num_sample_; ++i) {
      auto deal = resampler_->Resample(state);
      if (deal[0] == -1) {
        --i;
        continue;
      }
//      std::cout << "sampled deal " << i << std::endl;
      auto sampled_state = ConstructStateFromDeal(deal, state.ParentGame(), state);
//      std::cout << sampled_state->ToString() << std::endl;
      for (int j = 0; j < num_legal_moves; ++j) {
        int score = Rollout(*sampled_state, legal_moves[j]);
//        std::cout << score << std::endl;
        res.scores[j] += score;
      }
//      std::cout << "accumulate scores" << std::endl;

    }
    return res;
  }


 private:
  std::shared_ptr<Resampler> resampler_;
  int num_sample_;

};

void PrintSearchResult(const SearchResult &res) {
  for (int i = 0; i < res.moves.size(); ++i) {
    std::cout << "Move " << res.moves[i].ToString() << ", Score: " << res.scores[i] << "\n";
  }
}

#endif //BRIDGE_LEARNING_PLAYCC_PIMC_H_
