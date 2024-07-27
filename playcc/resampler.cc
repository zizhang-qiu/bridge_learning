//
// Created by qzz on 2023/11/20.
//
#include "resampler.h"

#include "bridge_lib/utils.h"

std::tuple<ResampleConstraints, std::vector<ble::BridgeHand>>
GetKnownCardsAndConstraintsFromState(const ble::BridgeState& state) {
  ResampleConstraints constraints{};
  std::fill(constraints.begin(), constraints.end(),
            std::array<int, ble::kNumSuits>{-1, -1, -1, -1});
  const std::vector<ble::BridgeHand> hands = state.OriginalDeal();
  const auto& history = state.History();
  const auto& play_history = GetPlayHistory(history);
  const ble::Contract contract = state.GetContract();
  const ble::Player current_player = state.CurrentPlayer();

  ble::Player player = (1 + contract.declarer) % ble::kNumPlayers;
  ble::Trick trick{ble::kInvalidPlayer, ble::kNoTrump, 0};
  std::vector<ble::BridgeHand> known_cards(ble::kNumPlayers);

  for (int i = 0; i < play_history.size(); ++i) {
    if (i % ble::kNumPlayers == 0) {
      if (i > 0) {
        player = trick.Winner();
      }
    } else {
      player = (1 + player) % ble::kNumPlayers;
    }
    const auto item = play_history[i];
    const auto card = ble::CardIndex(item.suit, item.rank);
    known_cards[player].AddCard({item.suit, item.rank});
    // A new trick
    if (i % ble::kNumPlayers == 0) {
      trick = ble::Trick(player, contract.denomination, card);
    } else {
      trick.Play(player, card);
      // This only happen when this player has run out of led suit.
      if (item.suit != trick.LedSuit()) {
        constraints[player][trick.LedSuit()] =
            known_cards[player].SuitLength()[trick.LedSuit()];
      }
    }
  }
  // Player knows his own cards.
  known_cards[current_player] = hands[current_player];
  if (state.IsDummyCardShown()) {
    const ble::Player dummy = state.GetDummy();
    known_cards[dummy] = hands[dummy];
  }
  //  std::cout << "Get current player's hand" << std::endl;
  return std::make_tuple(constraints, known_cards);
}

std::vector<std::array<int, ble::kNumCards>> ResampleMultipleDeals(
    const std::shared_ptr<Resampler>& resampler, const ble::BridgeState& state,
    int num_deals) {
  std::vector<std::array<int, ble::kNumCards>> deals;
  int num_sampled_deals = 0;
  while (num_sampled_deals < num_deals) {
    const ResampleResult result = resampler->Resample(state);
    if (result.success) {
      deals.push_back(result.result);
      ++num_sampled_deals;
    }
  }
  SPIEL_CHECK_EQ(num_sampled_deals, num_deals);
  return deals;
}

ResampleResult UniformResampler::Resample(const ble::BridgeState& state) {
  deck_sampler_.Reset();

  const ble::Player current_player = state.CurrentPlayer();
  //    const ble::Contract contract = state.GetContract();
  //    std::cout << "contract: " << contract.ToString() << std::endl;
  ResampleConstraints constraints{};
  std::vector<ble::BridgeHand> known_cards;
  std::tie(constraints, known_cards) =
      GetKnownCardsAndConstraintsFromState(state);
  //    std::cout << "constraints:\n";
  //    for (int i = 0; i < ble::kNumPlayers; ++i) {
  //      for (int j = 0; j < ble::kNumSuits; ++j) {
  //        std::cout << constraints[i][j] << ", ";
  //      }
  //      std::cout << "\n";
  //    }

  deck_sampler_.DealKnownCards(known_cards);
  //    std::cout << "known cards dealt." << std::endl;
  std::vector<ble::Player> player_needs_filter;
  std::vector<ble::Player> player_not_need_filter;
  std::vector<std::vector<ble::Suit>> suit_filters(ble::kNumPlayers);
  for (ble::Player pl : ble::kAllSeats) {
    std::vector<ble::Suit> suit_filter;
    for (ble::Suit suit : ble::kAllSuits) {
      if (constraints[pl][suit] != -1) {
        suit_filter.push_back(suit);
      }
    }
    if (suit_filter.empty()) {
      player_not_need_filter.push_back(pl);
    } else {
      player_needs_filter.push_back(pl);
    }
    suit_filters[pl] = suit_filter;
  }
  //    std::cout << "player needs filter\n";
  //    rela::utils::printVector(player_needs_filter);
  //    std::cout << "player not need filter\n";
  //    rela::utils::printVector(player_not_need_filter);

  auto sampled_hands = known_cards;
  for (const ble::Player pl : player_needs_filter) {
    // Don't need to sample our cards.
    if (pl == current_player) {
      continue;
    }

    const int num_cards_need = ble::kNumCardsPerHand -
                               static_cast<int>(known_cards[pl].Cards().size());
    for (int i = 0; i < num_cards_need; ++i) {
      ble::BridgeCard sampled_card;
      sampled_card = deck_sampler_.SampleNotSuits(suit_filters[pl], rng_);
      if (sampled_card.IsValid()) {
        sampled_hands[pl].AddCard(sampled_card);
      } else {
        return {false, {}};
      }
    }
  }

  for (ble::Player pl : player_not_need_filter) {
    // Don't need to sample our cards.
    if (pl == current_player) {
      continue;
    }

    const int num_cards_need = ble::kNumCardsPerHand -
                               static_cast<int>(known_cards[pl].Cards().size());
    //      std::cout << "num cards need: " << num_cards_need << std::endl;
    for (int i = 0; i < num_cards_need; ++i) {
      ble::BridgeCard sampled_card = deck_sampler_.Sample(rng_);
      if (sampled_card.IsValid()) {
        sampled_hands[pl].AddCard(sampled_card);
      } else {
        return {false, {}};
      }
    }
  }
  //    std::cout << "Sample done." << std::endl;
  //    for(ble::Player pl:ble::kAllSeats){
  //      for(const auto card:sampled_hands[pl].Cards()){
  //        std::cout << card.ToString() << ", ";
  //      }
  //      std::cout << "\n";
  //    }
  return {true, HandsToCardIndices(sampled_hands)};
}

void UniformResampler::ResetWithParams(
    const std::unordered_map<std::string, std::string>& params) {
  const auto seed = ble::ParameterValue<int>(params, "seed", 42);
  rng_.seed(seed);
}
