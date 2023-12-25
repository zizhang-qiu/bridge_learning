//
// Created by qzz on 2023/11/20.
//
#include "pimc.h"
#include "absl/strings/str_cat.h"

int Rollout(const ble::BridgeState& state, const ble::BridgeMove& move) {
  SetMaxThreads(0);
  auto child = state.Child(move);
  const ble::Contract contract = state.GetContract();
  if (child.IsTerminal()) {
    // The state may reach terminal after playing a card.
    if (IsActingPlayerDeclarerSide(state)) {
      return child.NumDeclarerTricks() >= contract.level + 6;
    }
    return child.NumDeclarerTricks() < contract.level + 6;
  }
  auto dl = StateToDDSDeal(child);

  const ble::Player child_player = child.CurrentPlayer();

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
    SpielFatalError(absl::StrCat("double dummy solver:", error_message));
  }

  const int num_tricks_left = ble::kNumTricks - child.NumTricksPlayed();

  // The player act at original state is declarer side.
  if (const bool is_max_node = IsActingPlayerDeclarerSide(state); is_max_node) {
    // The player act at child state is declarer side.
    if (ble::Partnership(child_player) == ble::Partnership(state.CurrentPlayer())) {
      return fut.score[0] + child.NumDeclarerTricks() >= (contract.level + 6);
    }
    else {
      // The player act at child state is defender side.
      return num_tricks_left - fut.score[0] + child.NumDeclarerTricks() >= (contract.level + 6);
    }
  }

  // The player act at original state is defender side.
  if (ble::Partnership(child_player) == ble::Partnership(state.CurrentPlayer())) {
    // The player act at child state is defender side.
    // Defender side wins if declarer win less tricks than (target level + 6)
    const int num_tricks_declarer_can_win = num_tricks_left - fut.score[0] + child.NumDeclarerTricks();
    return num_tricks_declarer_can_win < (contract.level + 6);
  }
  // The player act at child state is declarer side.
  const int num_tricks_declarer_can_win = fut.score[0] + child.NumDeclarerTricks();
  return num_tricks_declarer_can_win < (contract.level + 6);
}
std::pair<ble::BridgeMove, int> GetBestAction(const SearchResult& res) {
  const auto it = std::max_element(res.scores.begin(), res.scores.end());
  const int index = static_cast<int>(std::distance(res.scores.begin(), it));
  return std::make_pair(res.moves[index], res.scores[index]);
}
SearchResult PIMCBot::Search(const ble::BridgeState& state) const {
//  const auto legal_moves = state.LegalMoves();
  const auto legal_moves = GetLegalMovesWithoutEquivalentCards(state);
  const int num_legal_moves = static_cast<int>(legal_moves.size());
  //    std::cout << "num legal moves: " << num_legal_moves << std::endl;
  SearchResult res{};
  res.moves = legal_moves;
  res.scores = std::vector<int>(num_legal_moves, 0);

  //  if (num_legal_moves == 1) {
  //    return res;
  //  }
  //  for (int i = 0; i < num_sample_; ++i) {
  //    auto resample_result = resampler_->Resample(state);
  //    if (!resample_result.success) {
  //      --i;
  //      continue;
  //    }
  //    //      std::cout << "sampled deal " << i << std::endl;
  //    auto sampled_state = ConstructStateFromDeal(resample_result.result, state.ParentGame(), state);
  //    //      std::cout << sampled_state->ToString() << std::endl;
  //    for (int j = 0; j < num_legal_moves; ++j) {
  //      const int score = Rollout(sampled_state, legal_moves[j]);
  //      //        std::cout << score << std::endl;
  //      res.scores[j] += score;
  //    }
  //    //      std::cout << "accumulate scores" << std::endl;
  //  }
  const auto deals = ResampleMultipleDeals(resampler_, state, cfg_.num_worlds);
//  std::cout << "Deals sampled in pimc:\n" << std::endl;
//  for (const auto d : deals) {
//    PrintArray(d);
//  }
  for (int i = 0; i < cfg_.num_worlds; ++i) {
    auto sampled_state = ConstructStateFromDeal(deals[i], state.ParentGame(), state);
//    std::cout << sampled_state.ToString() << std::endl;
    for (int j = 0; j < num_legal_moves; ++j) {
      const int score = Rollout(sampled_state, legal_moves[j]);
      //        std::cout << score << std::endl;
      res.scores[j] += score;
    }
  }
  return res;
}
ble::BridgeMove PIMCBot::Act(const ble::BridgeState &state) {
  SPIEL_CHECK_EQ(static_cast<int>(state.CurrentPhase()), static_cast<int>(ble::Phase::kPlay));
//  const auto legal_moves = state.LegalMoves();
  const auto legal_moves = GetLegalMovesWithoutEquivalentCards(state);
  if (const int num_legal_moves = static_cast<int>(legal_moves.size()); num_legal_moves == 1){
    if(!cfg_.search_with_one_legal_move){
      return legal_moves[0];
    }
  }
  const SearchResult res = Search(state);
  auto [move, score] = GetBestAction(res);

  PrintSearchResult(res);

  return move;
}
void PrintSearchResult(const SearchResult& res) {
  for (int i = 0; i < res.moves.size(); ++i) {
    std::cout << "Move " << res.moves[i].ToString() << ", Score: " << res.scores[i] << "\n";
  }
}
