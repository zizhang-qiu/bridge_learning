//
// Created by qzz on 2024/1/16.
//

#include "torch_actor_resampler.h"

template <typename Container>
bool CheckDealLegality(const Container& cards) {
  Container cards_copy = cards;
  std::sort(cards_copy.begin(), cards_copy.end());
  for (int i = 0; i < ble::kNumCards; ++i) {
    if (cards_copy[i] != i) {
      return false;
    }
  }
  return true;
}

ResampleResult TorchActorResampler::Resample(const ble::BridgeState& state) {
  rela::TensorDict belief;
  // std::cout <<"Enter resample" << std::endl;
  if (state == state_) {
    // If state equals with cached state, reuse belief.
    belief = belief_;
  } else {
    const auto obs = MakeTensorDictObs(state);
    belief = torch_actor_->GetBelief(obs);
    belief_ = belief;
    state_ = state;
  }

  // std::cout << "enter sample." << std::endl;
  std::array<int, ble::kNumCards> deal{};
  while (true) {
    deal = SampleFromBelief(belief, state);
    if (deal[0] != -1) {
      break;
    }
  }
  // std::cout <<"finish sample." << std::endl;


  const auto game = state.ParentGame();
  auto sample_state = ConstructStateFromDeal(deal, state.ParentGame());

  const auto bidding_history = state.AuctionHistory();

  // std::cout << "enter filter." << std::endl;
  // Loop over bidding actions and filter.
  for (const auto& item : bidding_history) {
    // std::cout << item.ToString() << std::endl;
    const auto this_obs = MakeTensorDictObs(sample_state);
    const auto policy = torch_actor_->GetPolicy(this_obs);
    const auto action_uid = game->GetMoveUid(item.move);
    // std::cout << "action uid: " << action_uid << std::endl;
    const auto action_prob = policy.at("pi")[
      action_uid - ble::kBiddingActionBase].item<float>();
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    // Generate a random value.
    const float random_value = distribution(rng_);
    // std::cout <<"prob: " << action_prob << ", random: " << random_value << std::endl;
    if (action_prob < random_value) {
      return {false, deal};
    }
    sample_state.ApplyMove(item.move);
  }
  // std::cout << "Deal: \n";
  // for(const int a:deal) {
  //   std::cout << a << ", ";
  // }
  // std::cout << std::endl;
  return {true, deal};
}

void TorchActorResampler::ResetWithParams(
    const std::unordered_map<std::string, std::string>& params) {
  const auto seed = ble::ParameterValue<int>(params, "seed", 42);
  rng_.seed(seed);
}

rela::TensorDict TorchActorResampler::MakeTensorDictObs(
    const ble::BridgeState& state) const {
  const auto observation = ble::BridgeObservation(state, state.CurrentPlayer());
  auto encoding = encoder_.Encode(observation);
  encoding = {encoding.begin(),
              encoding.begin() + encoder_.GetAuctionTensorSize()};
  const auto& legal_moves = observation.LegalMoves();
  const auto game = state.ParentGame();

  if (state.CurrentPhase() == ble::Phase::kPlay) {
    rela::TensorDict obs = {
        {"s", torch::tensor(encoding, {torch::kFloat32})},
        {"legal_move", torch::ones(ble::kNumCalls, {torch::kFloat32})}
    };
    return obs;
  }
  std::vector<float> legal_move_mask(ble::kNumCalls, 0);
  for (const auto& move : legal_moves) {
    const int uid = game->GetMoveUid(move);
    legal_move_mask[uid - ble::kBiddingActionBase] = 1;
  }
  rela::TensorDict obs = {
      {"s", torch::tensor(encoding, {torch::kFloat32})},
      {"legal_move", torch::tensor(legal_move_mask, {torch::kFloat32})}
  };
  return obs;
}

std::array<int, ble::kNumCards> TorchActorResampler::SampleFromBelief(
    const rela::TensorDict& belief,
    const ble::BridgeState& state) const {
  const auto belief_probs = belief.at("belief");
  const torch::Tensor basic_indices =
      torch::arange(0, ble::kNumCardsPerHand) * ble::kNumPlayers;
  // std::cout <<"basic_indices:\n" << basic_indices << std::endl;
  const int observation_tensor_size = encoder_.GetAuctionTensorSize();

  const auto player_cards_feature = torch::tensor(
      encoder_.EncodeMyHand({state}));
  // std::cout << "player_cards_feature:\n" << player_cards_feature << std::endl;
  const auto player_cards = torch::nonzero(player_cards_feature).squeeze().to(
      torch::kInt32);
  // std::cout << "player_cards:\n" << player_cards << std::endl;
  // Cards have been selected
  torch::Tensor deal_cards = torch::ones(ble::kNumCards, {torch::kInt32}).
      fill_(-1);
  // std::cout <<"deal cards:\n" << deal_cards << std::endl;
  // std::cout << "current player: " << state.CurrentPlayer() <<std::endl;
  deal_cards = deal_cards.scatter_(0, basic_indices + state.CurrentPlayer(),
                                   player_cards);
  // std::cout << "deal cards:\n" << deal_cards << std::endl;
  torch::Tensor selected_cards = player_cards_feature.clone();

  for (const int sample_relative_player : {2, 1, 3}) {
    const int start_index = (sample_relative_player - 1) * ble::kNumCards;
    const int end_index = sample_relative_player * ble::kNumCards;
    torch::Tensor relative_player_pred = belief_probs.slice(
        0, start_index, end_index).clone();
  // std::cout << "relative player: " << sample_relative_player << ":\n" <<
  //     relative_player_pred << std::endl;
    relative_player_pred *= 1 - selected_cards;
    if (torch::count_nonzero(relative_player_pred).item<int>() <
        ble::kNumCardsPerHand) {
      return {-1}; // sentinel
    }
    // Sample 13 cards
    torch::Tensor sample_cards = torch::multinomial(
        relative_player_pred, ble::kNumCardsPerHand, false);
    selected_cards = selected_cards.scatter_(0, sample_cards, 1);
    // std::cout << selected_cards << std::endl;
    deal_cards = deal_cards.scatter_(0,
                                     basic_indices + (
                                       state.CurrentPlayer() +
                                       sample_relative_player) %
                                     ble::kNumPlayers,
                                     sample_cards.to(torch::kInt32));
    // std::cout << "deal cards:\n" << deal_cards << std::endl;
  }
  std::array<int, ble::kNumCards> ret{};
  std::copy_n(deal_cards.data_ptr<int>(),
              ble::kNumCards, ret.begin());
  // for(int i=0; i< ble::kNumCards; ++i) {
  //   std::cout << ret[i] << std::endl;
  // }
  return ret;
  // const bool is_legal = CheckDealLegality(ret);
  // return is_legal ? ret : std::array<int, ble::kNumCards>{-1};
}
