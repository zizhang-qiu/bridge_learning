#include "canonical_encoder.h"

#include <numeric>
#include <sstream>

#include "bridge_observation.h"

namespace bridge_learning_env {
// Computes the product of dimensions in shape, i.e. how many individual
// pieces of data the encoded observation requires.
int FlatLength(const std::vector<int>& shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
}

int EncodeVulnerabilityBoth(const BridgeObservation& obs,
                            const std::shared_ptr<BridgeGame>& game,
                            const int start_offset,
                            std::vector<int>* encoding) {
  int offset = start_offset;
  (*encoding)[offset + obs.IsPlayerVulnerable()] = 1;
  offset += kNumVulnerabilities;
  (*encoding)[offset + obs.IsOpponentVulnerable()] = 1;
  offset += kNumVulnerabilities;
  REQUIRE_EQ(offset - start_offset, kNumPartnerships * kNumVulnerabilities);
  return offset - start_offset;
}

int EncodeAuction(const BridgeObservation& obs, const int start_offset, std::vector<int>* encoding) {
  int offset = start_offset;
  const auto& history = obs.AuctionHistory();
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
    const auto& item = history[idx];
    if (item.other_call == kDouble) {
      (*encoding)[offset + (last_bid - kFirstBid) * kSingleBidTensorSize + kNumPlayers + item.player] = 1;
    }
    else if (item.other_call == kRedouble) {
      (*encoding)[offset + (last_bid - kFirstBid) * kSingleBidTensorSize + kNumPlayers * 2 + item.player] = 1;
    }
    else if (item.move.IsBid()) {
      // Should be a bid.
      const int bid_index = BidIndex(item.level, item.denomination);
      (*encoding)[offset + (bid_index - kFirstBid) * kSingleBidTensorSize + item.player] = 1;
      last_bid = bid_index;
    }
  }
  offset += kBiddingHistoryTensorSize;
  REQUIRE_EQ(offset - start_offset, kOpeningPassTensorSize + kBiddingHistoryTensorSize);
  return offset - start_offset;
}

int EncodePlayerHand(const BridgeObservation& obs, const int start_offset, std::vector<int>* encoding) {
  int offset = start_offset;
  const auto& cards = obs.Hands()[0].Cards();
  REQUIRE(cards.size() <= kNumCardsPerHand);
  for (const BridgeCard& card : cards) {
    REQUIRE(card.IsValid());
    (*encoding)[offset + card.Index()] = 1;
  }
  offset += kNumCards;
  return offset - start_offset;
}

// Convert double status to index, undoubled=0, doubled=1, redoubled=2.
int DoubleStausToIndex(const DoubleStatus& double_status) {
  switch (double_status) {
    case kUndoubled:
      return 0;
    case kDoubled:
      return 1;
    case kRedoubled:
      return 2;
    default:
      std::cerr << "Invalid double status: " << double_status << std::endl;
      std::abort();
  }
}

int EncodeContract(const BridgeObservation& obs, const int start_offset, std::vector<int>* encoding) {
  int offset = start_offset;

  // Contract level.
  const auto contract = obs.GetContract();
  (*encoding)[offset + contract.level - 1] = 1;
  offset += kNumBidLevels;

  // Contract denomination.
  (*encoding)[offset + contract.denomination] = 1;
  offset += kNumDenominations;

  // Double status.
  (*encoding)[offset + DoubleStausToIndex(contract.double_status)] = 1;
  offset += kNumDoubleStatus;

  // Declarer.
  const Player relative_declarer = (contract.declarer + kNumPlayers - obs.ObservingPlayer()) % kNumPlayers;
  (*encoding)[offset + relative_declarer] = 1;
  offset += kNumPlayers;
  REQUIRE_EQ(offset - start_offset, kNumBidLevels + kNumDenominations);
  return offset - start_offset;
}

int EncodeVulnerabilityDeclarer(const BridgeObservation& obs, const int start_offset, std::vector<int>* encoding) {
  int offset = start_offset;
  const bool is_observing_player_declarer = Partnership(obs.ObservingPlayer()) == Partnership(
    obs.GetContract().declarer);
  const bool vul = is_observing_player_declarer ? obs.IsPlayerVulnerable() : obs.IsOpponentVulnerable();

  (*encoding)[offset + vul] = 1;
  offset += kNumVulnerabilities;
  REQUIRE_EQ(offset - start_offset, kNumVulnerabilities);
  return offset - start_offset;
}

int EncodeDummyHand(const BridgeObservation& obs, const int start_offset, std::vector<int>* encoding) {
  int offset = start_offset;
  const int dummy = Partner(obs.GetContract().declarer);
  const int relative_dummy = PlayerToOffset(dummy, obs.ObservingPlayer());
  const auto& hands = obs.Hands();
  const auto& dummy_hand = hands[relative_dummy];

  for (const auto& card : dummy_hand.Cards()) {
    (*encoding)[offset + card.Index()] = 1;
  }
  offset += kNumCards;
  REQUIRE_EQ(offset - start_offset, kNumCards);
  return offset - start_offset;
}

int EncodePlayedTricks(const BridgeObservation& obs,
                       const int start_offset,
                       std::vector<int>* encoding,
                       const int num_tricks) {
  int offset = start_offset;

  const int current_trick = obs.NumCardsPlayed() / kNumPlayers;
  const int this_trick_cards_played = obs.NumCardsPlayed() % kNumPlayers;
  const int this_trick_start = obs.NumCardsPlayed() - this_trick_cards_played;
  const auto tricks = obs.Tricks();
  const auto& play_history = obs.PlayHistory();

  // Current trick.
  if (obs.CurrentPhase() != Phase::kGameOver) {
    int leader = tricks[current_trick].Leader();
    for (int i = 0; i < this_trick_cards_played; ++i) {
      const auto item = play_history[this_trick_cards_played + i];
      const int relative_player = PlayerToOffset(item.player, obs.ObservingPlayer());
      const int card_index = CardIndex(item.suit, item.rank);
      (*encoding)[offset + relative_player * kNumCards + card_index] = 1;
    }
  }

  offset += kNumCards * kNumPlayers;

  // Previous tricks.
  for (int j = current_trick - 1; j >= std::max(0, current_trick - num_tricks + 1); --j) {
    for (int i = 0; i < kNumPlayers; ++i) {
      const auto item = play_history[this_trick_start - kNumPlayers * (current_trick - j) + i];
      const int relative_player = PlayerToOffset(item.player, obs.ObservingPlayer());
      const int card_index = CardIndex(item.suit, item.rank);
      (*encoding)[offset + relative_player * kNumCards + card_index] = 1;
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

int EncodeNumTricksWon(const BridgeObservation& obs, const int start_offset, std::vector<int>* encoding) {
  int offset = start_offset;

  // Tricks won by declarer side.
  (*encoding)[offset + obs.NumDeclarerTricks()] = 1;
  offset += kNumTricks;

  // Tricks won by defender side.
  const int num_defender_tricks = obs.NumCardsPlayed() / 4 - obs.NumDeclarerTricks();
  (*encoding)[offset + num_defender_tricks] = 1;
  offset += kNumTricks;

  REQUIRE_EQ(offset-start_offset, kNumPartnerships * kNumTricks);
  return offset - start_offset;
}

std::vector<int> CanonicalEncoder::Shape() const {
  return {std::max(kBiddingTensorSize, GetPlayTensorSize())};
}

std::vector<int> CanonicalEncoder::Encode(const BridgeObservation& obs) const {
  std::vector<int> encoding(FlatLength(Shape()), 0);

  int offset = 0;

  // Play phase.
  if (obs.NumCardsPlayed() > 0) {
    offset += EncodeContract(obs, offset, &encoding);
    offset += EncodeVulnerabilityDeclarer(obs, offset, &encoding);
    offset += EncodePlayerHand(obs, offset, &encoding);
    offset += EncodeDummyHand(obs, offset, &encoding);
    offset += EncodePlayedTricks(obs, offset, &encoding, num_tricks_in_observation_);
    offset += EncodeNumTricksWon(obs, offset, &encoding);
  }
  else {
    offset += EncodeVulnerabilityBoth(obs, parent_game_, offset, &encoding);
    offset += EncodeAuction(obs, offset, &encoding);
    offset += EncodePlayerHand(obs, offset, &encoding);
  }

  REQUIRE(offset <= encoding.size());
  return encoding;
}
} // namespace bridge_learning_env
