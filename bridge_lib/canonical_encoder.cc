#include "canonical_encoder.h"

#include <numeric>

#include "bridge_observation.h"

namespace bridge_learning_env {
// Computes the product of dimensions in shape, i.e. how many individual
// pieces of data the encoded observation requires.
int FlatLength(const std::vector<int> &shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<>());
}

int VulnerabilitySectionLength() {
  return kNumVulnerabilities * kNumPartnerships;
}

int HandSectionLength() {
  return kNumCards * kNumPlayers;
}

int OpeningPassSectionLength() {
  return kNumPlayers;
}

int BiddingSectionLength() {
  return kSingleBidTensorSize * kNumBids;
}

int EncodeVulnerabilityBoth(const BridgeObservation &obs,
                            const std::shared_ptr<BridgeGame> &game,
                            const int start_offset,
                            std::vector<int> *encoding) {
  int offset = start_offset;
  (*encoding)[offset + obs.IsPlayerVulnerable()] = 1;
  offset += kNumVulnerabilities;
  (*encoding)[offset + obs.IsOpponentVulnerable()] = 1;
  offset += kNumVulnerabilities;
  REQUIRE_EQ(offset - start_offset, kNumPartnerships * kNumVulnerabilities);
  return offset - start_offset;
}

int EncodeAuction(const BridgeObservation &obs, const int start_offset,
                  std::vector<int> *encoding) {
  int offset = start_offset;
  const auto &history = obs.AuctionHistory();
  int idx = 0;
  // Opening pass
  for (; idx < history.size(); idx++) {
    if (history[idx].move.IsBid()) {
      break;
    }
    if (history[idx].move.OtherCall() == kPass) {
      (*encoding)[offset + history[idx].player] = 1;
    }
  }
  //  std::cout << "idx: " << idx << "\n";
  offset += kNumPlayers;
  // For each bid, a 4 * 3 = 12 bits block is used to track whether a player
  // makes/doubles/redoubles the bid.
  int last_bid = 0;
  for (; idx < history.size(); idx++) {
    const auto &item = history[idx];
    if (item.other_call == kDouble) {
      (*encoding)[offset + (last_bid - kFirstBid) * kSingleBidTensorSize +
          kNumPlayers + item.player] = 1;
    } else if (item.other_call == kRedouble) {
      (*encoding)[offset + (last_bid - kFirstBid) * kSingleBidTensorSize +
          kNumPlayers * 2 + item.player] = 1;
    } else if (item.move.IsBid()) {
      // Should be a bid.
      const int bid_index = BidIndex(item.level, item.denomination);
      (*encoding)[offset + (bid_index - kFirstBid) * kSingleBidTensorSize +
          item.player] = 1;
      last_bid = bid_index;
    }
  }
  offset += kBiddingHistoryTensorSize;
  REQUIRE_EQ(offset - start_offset,
             kOpeningPassTensorSize + kBiddingHistoryTensorSize);
  return offset - start_offset;
}

int EncodePlayerHand(const BridgeObservation &obs, const int start_offset,
                     std::vector<int> *encoding, const int relative_player) {
  int offset = start_offset;
  const auto &cards = obs.Hands()[relative_player].Cards();
  REQUIRE(cards.size() <= kNumCardsPerHand);
  for (const BridgeCard &card : cards) {
    REQUIRE(card.IsValid());
    (*encoding)[offset + card.Index()] = 1;
  }
  offset += kNumCards;
  return offset - start_offset;
}

int EncodeAllHands(const BridgeObservation &obs, const int start_offset,
                   std::vector<int> *encoding, const bool show_all_hands) {
  int offset = start_offset;
  // My hand.
  offset += EncodePlayerHand(obs, offset, encoding, /*relative_player=*/0);

  if (show_all_hands) {
    for (int player = 1; player < kNumPlayers; ++player) {
      offset += EncodePlayerHand(obs, offset, encoding, /*relative_player=*/player);
    }
  } else {
    offset += (kNumPlayers - 1) * kNumCards;
  }

  REQUIRE_EQ(offset - start_offset, kNumPlayers * kNumCards);
  return offset - start_offset;
}

// Convert double status to index, undoubled=0, doubled=1, redoubled=2.
int DoubleStatusToIndex(const DoubleStatus &double_status) {
  switch (double_status) {
    case kUndoubled:return 0;
    case kDoubled:return 1;
    case kRedoubled:return 2;
    default:std::cerr << "Invalid double status: " << double_status << std::endl;
      std::abort();
  }
}

int EncodeContract(const BridgeObservation &obs, const int start_offset,
                   std::vector<int> *encoding) {
  int offset = start_offset;

  // Contract level.
  const auto contract = obs.GetContract();
  (*encoding)[offset + contract.level - 1] = 1;
  offset += kNumBidLevels;

  // Contract denomination.
  (*encoding)[offset + contract.denomination] = 1;
  offset += kNumDenominations;

  // Double status.
  (*encoding)[offset + DoubleStatusToIndex(contract.double_status)] = 1;
  offset += kNumDoubleStatus;

  // Declarer.
  const Player relative_declarer =
      (contract.declarer + kNumPlayers - obs.ObservingPlayer()) % kNumPlayers;
  (*encoding)[offset + relative_declarer] = 1;
  offset += kNumPlayers;
  REQUIRE_EQ(offset - start_offset, kNumBidLevels + kNumDenominations +
      kNumDoubleStatus + kNumPlayers);
  return offset - start_offset;
}

int EncodeVulnerabilityDeclarer(const BridgeObservation &obs,
                                const int start_offset,
                                std::vector<int> *encoding) {
  int offset = start_offset;
  const bool is_observing_player_declarer =
      Partnership(obs.ObservingPlayer()) ==
          Partnership(obs.GetContract().declarer);
  const bool vul = is_observing_player_declarer ? obs.IsPlayerVulnerable()
                                                : obs.IsOpponentVulnerable();

  (*encoding)[offset + vul] = 1;
  offset += kNumVulnerabilities;
  REQUIRE_EQ(offset - start_offset, kNumVulnerabilities);
  return offset - start_offset;
}

int EncodeDummyHand(const BridgeObservation &obs, const int start_offset,
                    std::vector<int> *encoding) {
  int offset = start_offset;
  const int dummy = Partner(obs.GetContract().declarer);
  const int relative_dummy = PlayerToOffset(dummy, obs.ObservingPlayer());
  const auto &hands = obs.Hands();
  const auto &dummy_hand = hands[relative_dummy];

  for (const auto &card : dummy_hand.Cards()) {
    (*encoding)[offset + card.Index()] = 1;
  }
  offset += kNumCards;
  REQUIRE_EQ(offset - start_offset, kNumCards);
  return offset - start_offset;
}

int EncodePlayedTricks(const BridgeObservation &obs, const int start_offset,
                       std::vector<int> *encoding, const int num_tricks) {
  int offset = start_offset;

  const int current_trick = obs.NumCardsPlayed() / kNumPlayers;
  const int this_trick_cards_played = obs.NumCardsPlayed() % kNumPlayers;
  const int this_trick_start = obs.NumCardsPlayed() - this_trick_cards_played;
  const auto &play_history = obs.PlayHistory();

  // Current trick.
  if (obs.CurrentPhase() != Phase::kGameOver) {
    for (int i = 0; i < this_trick_cards_played; ++i) {
      const auto item = play_history[current_trick + i];
      // const int relative_player = PlayerToOffset(item.player, obs.ObservingPlayer());
      const int card_index = CardIndex(item.suit, item.rank);
      (*encoding)[offset + item.player * kNumCards + card_index] = 1;
    }
  }

  offset += kNumCards * kNumPlayers;

  // Previous tricks.
  for (int j = current_trick - 1;
       j >= std::max(0, current_trick - num_tricks + 1); --j) {
    for (int i = 0; i < kNumPlayers; ++i) {
      const auto item = play_history[this_trick_start -
          kNumPlayers * (current_trick - j) + i];
      // const int relative_player = PlayerToOffset(item.player, obs.ObservingPlayer());
      const int card_index = CardIndex(item.suit, item.rank);
      (*encoding)[offset + item.player * kNumCards + card_index] = 1;
    }
    offset += kNumPlayers * kNumCards;
  }

  // Future tricks.
  if (num_tricks > current_trick + 1) {
    offset += kNumPlayers * kNumCards * (num_tricks - current_trick - 1);
  }
  REQUIRE_EQ(offset - start_offset, num_tricks * kNumPlayers * kNumCards);
  return offset - start_offset;
}

int EncodeNumTricksWon(const BridgeObservation &obs, const int start_offset,
                       std::vector<int> *encoding) {
  int offset = start_offset;

  // Tricks won by declarer side.
  (*encoding)[offset + obs.NumDeclarerTricks()] = 1;
  offset += kNumTricks;

  // Tricks won by defender side.
  const int num_defender_tricks =
      obs.NumCardsPlayed() / 4 - obs.NumDeclarerTricks();
  (*encoding)[offset + num_defender_tricks] = 1;
  offset += kNumTricks;

  REQUIRE_EQ(offset - start_offset, kNumPartnerships * kNumTricks);
  return offset - start_offset;
}

int EncodeHandEvaluationOneHot(const BridgeObservation &obs, int start_offset,
                               std::vector<int> *encoding,
                               int relative_player) {
  int offset = start_offset;
  const auto &hand = obs.Hands()[relative_player];
  const auto hand_evaluation = hand.GetHandEvaluation();
  // HCP
  (*encoding)[offset + hand_evaluation.hcp] = 1;
  offset += kHCPTensorSize;
  // Control
  (*encoding)[offset + hand_evaluation.control] = 1;
  offset += kControlTensorSize;
  // Suit Length
  for (const Suit suit : kAllSuits) {
    (*encoding)[offset + hand_evaluation.suit_length[suit]] = 1;
    offset += kNumCardsPerSuit + 1;
  }

  REQUIRE_EQ(offset - start_offset, kHandEvaluationOneHotTensorSize);

  return offset - start_offset;
}

int EncodeHandEvaluation(const BridgeObservation &obs, int start_offset,
                         std::vector<int> *encoding, int relative_player) {
  int offset = start_offset;
  const auto &hand = obs.Hands()[relative_player];
  const auto hand_evaluation = hand.GetHandEvaluation();
  // HCP
  (*encoding)[offset] = hand_evaluation.hcp;
  offset += 1;
  // Control
  (*encoding)[offset] = hand_evaluation.control;
  offset += 1;
  // Suit Length
  for (const Suit suit : kAllSuits) {
    (*encoding)[offset] = hand_evaluation.suit_length[suit];
    offset += 1;
  }

  REQUIRE_EQ(offset - start_offset, kHandEvaluationTensorSize);

  return offset - start_offset;
}

std::vector<int> CanonicalEncoder::Shape() const {
  const int l = {
      VulnerabilitySectionLength()
          + OpeningPassSectionLength()
          + BiddingSectionLength()
          + HandSectionLength()
  };
  return {l};
}

std::vector<int> CanonicalEncoder::EncodeMyHand(
    const BridgeObservation &obs) const {
  std::vector<int> encoding(kNumCards, 0);
  int offset = 0;
  offset += EncodePlayerHand(obs, offset, &encoding, /*relative_player=*/0);
  REQUIRE_EQ(offset, kNumCards);
  return encoding;
}

std::vector<int> CanonicalEncoder::EncodeOtherHands(
    const BridgeObservation &obs) const {
  std::vector<int> encoding((kNumPlayers - 1) * kNumCards, 0);
  int offset = 0;
  for (int relative_player = 1; relative_player <= 3; ++relative_player) {
    offset += EncodePlayerHand(obs, offset, &encoding, relative_player);
  }
  REQUIRE_EQ(offset, (kNumPlayers - 1) * kNumCards);
  return encoding;
}

std::vector<int> CanonicalEncoder::EncodeOtherHandEvaluationsOneHot(
    const BridgeObservation &obs) const {
  std::vector<int> encoding((kNumPlayers - 1) * kHandEvaluationOneHotTensorSize,
                            0);
  int offset = 0;
  for (int relative_player = 1; relative_player <= 3; ++relative_player) {
    offset +=
        EncodeHandEvaluationOneHot(obs, offset, &encoding, relative_player);
  }
  REQUIRE_EQ(offset, (kNumPlayers - 1) * kHandEvaluationOneHotTensorSize);
  return encoding;
}

std::vector<int> CanonicalEncoder::EncodeOtherHandEvaluations(
    const BridgeObservation &obs) const {
  std::vector<int> encoding((kNumPlayers - 1) * kHandEvaluationTensorSize, 0);
  int offset = 0;
  for (int relative_player = 1; relative_player <= 3; ++relative_player) {
    offset += EncodeHandEvaluation(obs, offset, &encoding, relative_player);
  }
  REQUIRE_EQ(offset, (kNumPlayers - 1) * kHandEvaluationTensorSize);
  return encoding;
}

std::vector<int> CanonicalEncoder::Encode(const BridgeObservation &obs,
                                          const std::unordered_map<std::string, std::any> &kwargs) const {
  std::vector<int> encoding(FlatLength(Shape()), 0);

  int offset = 0;

  offset += EncodeVulnerabilityBoth(obs, parent_game_, offset, &encoding);
  offset += EncodeAuction(obs, offset, &encoding);
  bool show_all_hands = true;
  if (kwargs.count("show_all_hands")) {
    show_all_hands = std::any_cast<bool>(kwargs.at("show_all_hands"));
  }
  offset += EncodeAllHands(obs, offset, &encoding, /*show_all_hands=*/show_all_hands);

  REQUIRE(offset <= encoding.size());
  return encoding;
}

}  // namespace bridge_learning_env
