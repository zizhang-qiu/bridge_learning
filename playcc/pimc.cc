//
// Created by qzz on 2023/11/20.
//
#include "pimc.h"
int Rollout(const ble::BridgeState& state, const ble::BridgeMove& move) {
  auto cloned = state.Clone();
  cloned.ApplyMove(move);
  auto dl = StateToDDSDeal(cloned);
  futureTricks fut{};

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
  const int num_tricks_left = 13 - state.NumTricksPlayed();
  return state.NumDeclarerTricks() + num_tricks_left - fut.score[0] >= 6 + state.GetContract().level;
}
std::pair<ble::BridgeMove, int> GetBestAction(const SearchResult& res) {
  const auto it = std::max_element(res.scores.begin(), res.scores.end());
  const int index = static_cast<int>(std::distance(res.scores.begin(), it));
  return std::make_pair(res.moves[index], res.scores[index]);
}
SearchResult PIMCBot::Search(const ble::BridgeState& state) const {
  const auto legal_moves = state.LegalMoves();
  const int num_legal_moves = static_cast<int>(legal_moves.size());
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
      const int score = Rollout(sampled_state, legal_moves[j]);
      //        std::cout << score << std::endl;
      res.scores[j] += score;
    }
    //      std::cout << "accumulate scores" << std::endl;
  }
  return res;
}
void PrintSearchResult(const SearchResult& res) {
  for (int i = 0; i < res.moves.size(); ++i) {
    std::cout << "Move " << res.moves[i].ToString() << ", Score: " << res.scores[i] << "\n";
  }
}
