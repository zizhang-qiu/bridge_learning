//
// Created by qzz on 2023/10/22.
//

#ifndef BRIDGE_LEARNING_PLAYCC_PIMC_H_
#define BRIDGE_LEARNING_PLAYCC_PIMC_H_
#include <utility>

#include "bridge_lib/bridge_state_2.h"
#include "resampler.h"
#include "play_bot.h"
//#include "rela/logging.h"
#include "bridge_lib/third_party/dds/include/dll.h"
namespace ble = bridge_learning_env;


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
  int num_tricks_left = 13 - state.NumTricksPlayed();
  return num_tricks_left - fut.score[0];
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

class PIMCBot : public PlayBot{
 public:
  PIMCBot(std::shared_ptr<Resampler> resampler, int num_sample)
      : resampler_(std::move(resampler)), num_sample_(num_sample) {
    SetMaxThreads(0);
  }

  ble::BridgeMove Act(const ble::BridgeState2 &state) override{
    const SearchResult res = Search(state);
    auto [move, score] = GetBestAction(res);
    return move;
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
