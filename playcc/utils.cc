//
// Created by qzz on 2023/11/14.
//
#include "utils.h"

#include <algorithm>

#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/utils.h"

std::vector<ble::BridgeHistoryItem> GetPlayHistory(const std::vector<ble::BridgeHistoryItem> &history) {
  std::vector<ble::BridgeHistoryItem> play_history;
  for (const auto item : history) {
    if (item.move.MoveType() == ble::BridgeMove::Type::kPlay) {
      play_history.push_back(item);
    }
  }
  return play_history;
}
std::array<int, ble::kNumCards> HandsToCardIndices(const std::vector<ble::BridgeHand> &hands) {
  std::array<int, ble::kNumCards> res{};
  for (int i = 0; i < ble::kNumCardsPerHand; ++i) {
    for (int pl = 0; pl < ble::kNumPlayers; ++pl) {
      res[i * ble::kNumPlayers + pl] = hands[pl].Cards()[i].Index();
    }
  }
  return res;
}
ble::BridgeState ConstructStateFromDeal(const std::array<int, ble::kNumCards> &deal,
                                        const std::shared_ptr<ble::BridgeGame> &game) {
  auto state = ble::BridgeState(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state.ApplyMove(move);
  }
  return state;
}
ble::BridgeState ConstructStateFromDeal(const std::array<int, ble::kNumCards> &deal,
                                        const std::shared_ptr<ble::BridgeGame> &game,
                                        const ble::BridgeState &original_state) {
  auto state = ble::BridgeState(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state.ApplyMove(move);
  }
  const auto &history = original_state.History();
  for (int i = ble::kNumCards; i < history.size(); ++i) {
    ble::BridgeMove move = history[i].move;
    state.ApplyMove(move);
  }
  return state;
}
deal StateToDDSDeal(const ble::BridgeState &state) {
  // Should be play phase or game over.
  SPIEL_CHECK_EQ(static_cast<int>(state.CurrentPhase()), static_cast<int>(ble::Phase::kPlay));
  deal dl{};
  const ble::Contract contract = state.GetContract();
  // Trump.
  dl.trump = ble::DenominationToDDSStrain(contract.denomination);
  const ble::Trick current_trick = state.CurrentTrick();
  dl.first = current_trick.Leader() != ble::kInvalidPlayer ? current_trick.Leader()
      : state.IsDummyActing()                              ? state.GetDummy()
                                                           : state.CurrentPlayer();
  //  std::cout << "first: " << dl.first << std::endl;
  const auto play_history = state.PlayHistory();

  const int num_tricks_played = static_cast<int>(play_history.size()) / ble::kNumPlayers;
  const int num_card_played_current_trick =
      static_cast<int>(play_history.size()) - num_tricks_played * ble::kNumPlayers;
  memset(dl.currentTrickSuit, 0, 3 * sizeof(dl.currentTrickSuit));
  memset(dl.currentTrickRank, 0, 3 * sizeof(dl.currentTrickSuit));
  for (int i = 0; i < num_card_played_current_trick; ++i) {
    ble::BridgeHistoryItem item = play_history[num_tricks_played * ble::kNumPlayers + i];
    dl.currentTrickSuit[i] = ble::SuitToDDSSuit(item.suit);
    dl.currentTrickRank[i] = ble::RankToDDSRank(item.rank);
  }

  //  std::cout << "currentTrickSuit: ";
  //  for (int i : dl.currentTrickSuit) {
  //    std::cout << i << std::endl;
  //  }
  //
  //  std::cout << "currentTrickRank: ";
  //  for (int i : dl.currentTrickRank) {
  //    std::cout << i << std::endl;
  //  }

  const auto &hands = state.Hands();
  for (const ble::Player pl : ble::kAllSeats) {
    for (const auto card : hands[pl].Cards()) {
      dl.remainCards[pl][SuitToDDSSuit(card.CardSuit())] += 1 << 2 + card.Rank();
    }
  }
  return dl;
}

std::vector<int> MovesToUids(const std::vector<ble::BridgeMove> &moves, const ble::BridgeGame &game) {
  std::vector<int> uids;
  for (const auto &move : moves) {
    const int uid = game.GetMoveUid(move);
    uids.push_back(uid);
  }
  return uids;
}
bool IsActingPlayerDeclarerSide(const ble::BridgeState &state) {
  const auto declarer = state.GetContract().declarer;
  const auto cur_player = state.CurrentPlayer();
  return ble::Partnership(declarer) == ble::Partnership(cur_player);
}

std::array<std::vector<ble::BridgeCard>, ble::kNumSuits> SplitCardsVecBySuits(const vector<ble::BridgeCard> &cards) {
  std::array<std::vector<ble::BridgeCard>, ble::kNumSuits> rv{};
  for (const auto &card : cards) {
    rv[card.CardSuit()].push_back(card);
  }
  return rv;
}

std::set<ble::Suit> GetSuitsFromCardsVec(const std::vector<ble::BridgeCard> &cards) {
  std::set<ble::Suit> rv{};
  for (const auto &card : cards) {
    rv.emplace(card.CardSuit());
  }
  return rv;
}

std::set<ble::Suit> GetSuitsFromMovesVec(const std::vector<ble::BridgeMove> &moves) {
  std::set<ble::Suit> rv{};
  for (const auto &move : moves) {
    if (move.MoveType() == ble::BridgeMove::Type::kPlay) {
      rv.emplace(move.CardSuit());
    }
  }
  return rv;
}

std::vector<ble::BridgeCard> GenerateAllCardsBySuits(const std::set<ble::Suit> &suits) {
  std::vector<ble::BridgeCard> cards{};
  for (const ble::Suit suit : suits) {
    for (int rank = 0; rank < ble::kNumCardsPerSuit; ++rank) {
      cards.emplace_back(suit, rank);
    }
  }
  return cards;
}

std::vector<ble::BridgeCard> ExtractCardsBySuitsFromCardsVec(const std::vector<ble::BridgeCard> &cards,
                                                             const std::set<ble::Suit> &suits) {
  std::vector<ble::BridgeCard> rv{};
  for (const auto &card : cards) {
    ble::Suit card_suit = card.CardSuit();
    if (suits.find(card_suit) != suits.end()) {
      rv.push_back(card);
    }
  }
  return rv;
}

std::vector<ble::BridgeMove> GetLegalMovesWithoutEquivalentCards(const ble::BridgeState &state) {
  std::vector<ble::BridgeMove> moves;
  const auto legal_moves = state.LegalMoves();

  // What suits do we need to analyze?
  const auto suits = GetSuitsFromMovesVec(legal_moves);

  const auto legal_cards = ExtractCardsBySuitsFromCardsVec(
      state.Hands()[state.IsDummyActing() ? state.GetDummy() : state.CurrentPlayer()].Cards(),

      suits);
  const auto legal_cards_by_suits = SplitCardsVecBySuits(legal_cards);
  //  std::cout << "legal cards:" << std::endl;
  //  for (const auto &card : legal_cards) {
  //    std::cout << card.ToString() << std::endl;
  //  }

  const auto &played_cards = state.PlayedCards();
  //  const auto relevant_played_cards = ExtractCardsBySuitsFromCardsVec(played_cards, suits);
  const auto dummy_hand = state.Hands()[state.GetDummy()];
  const auto declarer_hand = state.Hands()[state.GetContract().declarer];

  const bool is_dummy_acting = state.IsDummyActing();

  // Get all cards we need to analyze.
  std::vector<ble::BridgeCard> all_cards = GenerateAllCardsBySuits(suits);
  //  std::vector<int> all_cards_indices = ble::Arange(0, ble::kNumCards);
  // First, erase the played cards.
  for (const auto &card : played_cards) {
    auto it = std::find(all_cards.begin(), all_cards.end(), card);
    if (it != all_cards.end()) {
      all_cards.erase(it);
    }
  }

  //  if (is_dummy_acting) {
  //    for (const auto &card : declarer_hand.Cards()) {
  //      auto it = std::find(all_cards.begin(), all_cards.end(), card);
  //      if (it != all_cards.end()) {
  //        all_cards.erase(it);
  //      }
  //    }
  //  } else {
  //    for (const auto &card : dummy_hand.Cards()) {
  //      auto it = std::find(all_cards.begin(), all_cards.end(), card);
  //      if (it != all_cards.end()) {
  //        all_cards.erase(it);
  //      }
  //    }
  //  }

  //  std::cout << "all cards:" << std::endl;
  //  for (const auto &card : all_cards) {
  //    std::cout << card.ToString() << std::endl;
  //  }

  auto cards_by_suits = SplitCardsVecBySuits(all_cards);
  auto card_value = [cards_by_suits](const ble::BridgeCard &card) -> int {
    const auto it = std::find(cards_by_suits[card.CardSuit()].begin(), cards_by_suits[card.CardSuit()].end(), card);
    return static_cast<int>(std::distance(cards_by_suits[card.CardSuit()].begin(), it));
  };

  std::array<std::vector<int>, ble::kNumCards> card_values_by_suits{};
  for (const ble::Suit suit : ble::kAllSuits) {
    for (const auto &card : cards_by_suits[suit]) {
      card_values_by_suits[suit].push_back(card_value(card));
    }
  }

  //  for (const ble::Suit suit : ble::kAllSuits) {
  //    for (const int value : card_values_by_suits[suit]) {
  //      std::cout << value << ", ";
  //    }
  //    std::cout << std::endl;
  //  }

  std::array<std::vector<int>, ble::kNumCards> legal_card_values_by_suits{};

  for (const ble::Suit suit : suits) {
    const auto &legal_cards_this_suit = legal_cards_by_suits[suit];
    const auto &cards_this_suit = cards_by_suits[suit];
    for (const auto &card : legal_cards_this_suit) {
      auto it = std::find(cards_this_suit.begin(), cards_this_suit.end(), card);
      size_t index = std::distance(cards_this_suit.begin(), it);
      int this_card_value = card_values_by_suits[suit][index];
      legal_card_values_by_suits[suit].push_back(this_card_value);
    }
    std::sort(legal_card_values_by_suits[suit].begin(), legal_card_values_by_suits[suit].end());
  }

  //  std::cout << "legal card values:" << std::endl;
  //  for (const ble::Suit suit : ble::kAllSuits) {
  //    for (const int value : legal_card_values_by_suits[suit]) {
  //      std::cout << value << ", ";
  //    }
  //    std::cout << std::endl;
  //  }

  std::array<std::vector<int>, ble::kNumCards> remained_legal_card_values_by_suits{};
  for (const ble::Suit suit : suits) {
    remained_legal_card_values_by_suits[suit] = KeepLargestConsecutive(legal_card_values_by_suits[suit]);
  }
  //  std::cout << "remained legal card values:" << std::endl;
  //  for (const ble::Suit suit : ble::kAllSuits) {
  //    for (const int value : remained_legal_card_values_by_suits[suit]) {
  //      std::cout << value << ", ";
  //    }
  //    std::cout << std::endl;
  //  }

  // Construct moves
  for (const ble::Suit suit : suits) {
    for (const int value : remained_legal_card_values_by_suits[suit]) {
      const auto it = std::find(card_values_by_suits[suit].begin(), card_values_by_suits[suit].end(), value);
      const size_t index = std::distance(card_values_by_suits[suit].begin(), it);
      const auto card = cards_by_suits[suit][index];
      const ble::BridgeMove move{ble::BridgeMove::Type::kPlay,
                                 card.CardSuit(),
                                 card.Rank(),
                                 ble::Denomination::kInvalidDenomination,
                                 -1,
                                 ble::OtherCalls::kNotOtherCall};
      moves.push_back(move);
    }
  }

  return moves;
}
std::vector<int> KeepLargestConsecutive(const std::vector<int> &input) {
  std::vector<int> result;

  if (input.empty()) {
    return result;
  }

  // Initialize the current consecutive sequence with the first element
  int current_consecutive = input[0];

  // Iterate through the input vector starting from the second element
  for (size_t i = 1; i < input.size(); ++i) {
    // Check if the current element is consecutive to the previous one
    if (input[i] == current_consecutive + 1) {
      // Update the current consecutive sequence to the larger value
      current_consecutive = input[i];
    }
    else {
      // If not consecutive, add the largest value in the current sequence to the result
      result.push_back(current_consecutive);

      // Update the current consecutive sequence to the current element
      current_consecutive = input[i];
    }
  }

  // Add the last element of the sequence to the result
  result.push_back(current_consecutive);

  return result;
}

std::vector<int> FindSetBitPositions(int decimalNumber) {
  std::vector<int> set_bit_positions;

  for (int position = 0; decimalNumber > 0; ++position) {
    if (decimalNumber & 1) {
      set_bit_positions.push_back(position);
    }
    decimalNumber >>= 1; // Right shift to check the next bit
  }

  return set_bit_positions;
}

std::vector<ble::BridgeMove> GetMovesFromFutureTricks(const futureTricks &fut) {
  std::vector<ble::BridgeMove> moves;
  for (int i = 0; i < ble::kNumCardsPerHand; ++i) {
    if (fut.rank[i] != 0) {
      moves.emplace_back(
          /*move_type=*/ble::BridgeMove::Type::kPlay,
          /*suit=*/ble::DDSSuitToSuit(fut.suit[i]),
          /*rank=*/ble::DDSRankToRank(fut.rank[i]),
          /*denomination=*/ble::kInvalidDenomination,
          /*level=*/-1,
          /*other_call=*/ble::kNotOtherCall);
    }

    // Deal with equal cards.
    if (fut.equals[i] != 0) {
      const std::vector<int> positions = FindSetBitPositions(fut.equals[i]);
      for (const int pos : positions) {
        moves.emplace_back(
            /*move_type=*/ble::BridgeMove::Type::kPlay,
            /*suit=*/ble::DDSSuitToSuit(fut.suit[i]),
            /*rank=*/ble::DDSRankToRank(pos),
            /*denomination=*/ble::kInvalidDenomination,
            /*level=*/-1,
            /*other_call=*/ble::kNotOtherCall);
      }
    }
  }
  return moves;
}
ble::BridgeState ConstructStateFromTrajectory(const std::vector<int> &trajectory,
                                              const std::shared_ptr<ble::BridgeGame> &game) {
  ble::BridgeState state{game};
  const int trajectory_length = static_cast<int>(trajectory.size());
  for (int i = 0; i < std::min(trajectory_length, ble::kNumCards); ++i) {
    const ble::BridgeMove move = game->GetChanceOutcome(trajectory[i]);
    state.ApplyMove(move);
  }
  for (int i = ble::kNumCards; i < trajectory_length; ++i) {
    const ble::BridgeMove move = game->GetMove(trajectory[i]);
    state.ApplyMove(move);
  }
  return state;
}
