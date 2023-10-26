//#define NOMINMAX
#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/bridge_observation.h"
#include "bridge_lib/example_cards_ddts.h"
//#include "rela/batch_runner.h"
//#include "rlcc/bridge_env.h"
#include <sstream>
#include <iostream>
#include "playcc/resampler.h"
#include "playcc/pimc.h"
#include "rela/utils.h"
namespace ble = bridge_learning_env;
int main() {
  const ble::GameParameters params = {};
  const auto game = std::make_shared<ble::BridgeGame>(params);
  auto state = ble::BridgeState2(game);

  // Deal
  const auto &deal = ble::example_deals[0];
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state.ApplyMove(move);
  }

  std::cout << state.ToString() << std::endl;

  std::vector<int> bid_uid;
  bid_uid.push_back(ble::BidIndex(1, ble::kSpadesTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(3, ble::kClubsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(3, ble::kDiamondsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(3, ble::kHeartsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(4, ble::kClubsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(4, ble::kHeartsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(5, ble::kClubsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(5, ble::kHeartsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kDouble + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  // Bidding
  for (const int uid : bid_uid) {
    ble::BridgeMove move = game->GetMove(uid);
    state.ApplyMove(move);
  }
  std::cout << state.ToString() << std::endl;

//  auto legal_moves = state.LegalMoves();
//  for(const auto& move:legal_moves){
//    std::cout << move.ToString() << std::endl;
//  }
//  state.ApplyMove(legal_moves[0]);

//  std::vector<int> scores(legal_moves.size(), 0);
//  for(int i=0; i<legal_moves.size(); ++i){
//    int score = Rollout(state, legal_moves[i]);
//    scores[i] = score;
//  }
//
//  rela::utils::printVector(scores);
  auto resampler = std::make_shared<UniformResampler>(21);
  PIMCBot bot{resampler, 100};
//  auto sampled_deal = resampler->Resample(state);
//  for(int i=0; i<ble::kNumCards; ++i){
//    std::cout << sampled_deal[i] << std::endl;
//  }
//  auto sampled_state = ConstructStateFromDeal(sampled_deal, game);
//  std::cout << sampled_state->ToString() << std::endl;
  while (!state.IsTerminal()) {
    auto legal_moves = state.LegalMoves();
//    std::cout << "legal moves:\n";
//    for (const auto &move : legal_moves) {
//      std::cout << move.ToString() << std::endl;
//    }
    auto res = bot.Search(state);
    PrintSearchResult(res);

//  for(const auto move:res.moves){
//    std::cout << move.ToString() << std::endl;
//  }
//  std::cout << res.scores.size() << std::endl;
//  rela::utils::printVector(res.scores);
    auto [move, score] = GetBestAction(res);
    std::cout << "Chosen move: " << move.ToString() << std::endl;
    state.ApplyMove(move);
    std::cout << state.ToString() << std::endl;
  }


//  std::vector<int> a ={39, 27, 15, 30, 41, 17, 44, 40, 36, 28, 12, 4, 0};
//  std::sort(a.begin(), a.end());
//  rela::utils::printVector(a);

}