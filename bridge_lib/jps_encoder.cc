#include "jps_encoder.h"
#include <vector>

#include "canonical_encoder.h"

namespace bridge_learning_env {

// In JPS encoding, the bidding history of players are ordered
// Own, partner, left opponent, right opponent.
int RelativePlayerToJPSPlayer(int relative_player) {
  if (relative_player == 0) {
    return relative_player;
  }
  // Partner = 2->1, left opponent = 1->2.
  return 3 - relative_player;
}

int EncodeJPSBiddingHistory(const BridgeObservation& obs, int start_offset,
                            std::vector<int>* encoding) {
  int offset = start_offset;

  const auto& bidding_history = obs.AuctionHistory();

  AuctionTracker auction_tracker_{};
  for (const auto& item : bidding_history) {
    // std::cout << item << std::endl;
    auction_tracker_.ApplyAuction(item.move, item.player);
    if (item.move.IsBid()) {
      const int index = (RelativePlayerToJPSPlayer(item.player) * kNumBids) +
                        (item.level - 1) * kNumDenominations +
                        item.denomination;
      (*encoding)[offset + index] = 1;
    } else if (item.move.OtherCall() == OtherCalls::kDouble) {
      const auto contract = auction_tracker_.GetContract();
      const int index = (kNumPlayers * kNumBids) +
                        (contract.level - 1) * kNumDenominations +
                        contract.denomination;
      (*encoding)[offset + index] = 1;
    }
  }
  offset += (kNumPlayers + 1) * kNumBids;
  REQUIRE_EQ(offset - start_offset, (kNumPlayers + 1) * kNumBids);
  return offset - start_offset;
}

int EncodeJPSVulnerability(const BridgeObservation& obs, int start_offset,
                           std::vector<int>* encoding) {
  int offset = start_offset;
  int cur_player_partnership = Partnership(obs.ObservingPlayer());
  const bool is_ns_vul = cur_player_partnership == 0
                             ? obs.IsPlayerVulnerable()
                             : obs.IsOpponentVulnerable();
  const bool is_ew_vul = cur_player_partnership == 1
                             ? obs.IsPlayerVulnerable()
                             : obs.IsOpponentVulnerable();
  (*encoding)[offset] = is_ns_vul;
  (*encoding)[offset + 1] = is_ew_vul;

  offset += kNumPartnerships;
  REQUIRE_EQ(offset - start_offset, kNumPartnerships);

  return offset - start_offset;
}

int EncodeJPSLegalMoves(const BridgeObservation& obs, int start_offset,
                        std::vector<int>* encoding) {
  int offset = start_offset;

  const auto& legal_moves = obs.LegalMoves();
  for (const auto& move : legal_moves) {
    const int index =
        move.IsBid() ? BidIndex(move.BidLevel(), move.BidDenomination()) - 3
                     : move.OtherCall() + 35;
    // std::cout << "move: "<< move << ", index: " << index << std::endl;
    (*encoding)[offset + index] = 1;
  }
  // Padding.
  (*encoding)[offset + 39 - 1] = 0;

  offset += kNumCalls + 1;
  REQUIRE_EQ(offset - start_offset, kNumCalls + 1);

  return offset - start_offset;
}

int EncodeJPSPlayerHand(const BridgeObservation& obs, int start_offset,
                        std::vector<int>* encoding, const int relative_player) {
  int offset = start_offset;
  // In JPS, card rank is AKQJT98765432, suit is CDHS
  for(const auto& card : obs.Hands()[relative_player].Cards()){
    const int card_index = (12 - card.Rank()) + card.CardSuit() * kNumCardsPerSuit;
    (*encoding)[offset + card_index] = 1;
  }

  offset += kNumCards;
  REQUIRE_EQ(offset - start_offset, kNumCards);
  return offset - start_offset;
}

std::vector<int> JPSEncoder::Encode(const BridgeObservation& obs,
                                    const std::unordered_map<std::string, std::any> &kwargs) const {
  REQUIRE(obs.CurrentPhase() == bridge_learning_env::Phase::kAuction);
  std::vector<int> encoding(kJPSTensorSize, 0);
  int offset = 0;
  // Hand.
  // std::cout << "Before encode hand." << std::endl;
  offset += EncodeJPSPlayerHand(obs, offset, &encoding,
                             /*relative_player=*/0);
  // Bidding history.
  // std::cout << "Before encode history." << std::endl;
  offset += EncodeJPSBiddingHistory(obs, offset, &encoding);

  // Vulnerability.
  // std::cout << "Before encode vul." << std::endl;
  offset += EncodeJPSVulnerability(obs, offset, &encoding);

  // Legal moves.
  // std::cout << "Before encode legal move." << std::endl;
  offset += EncodeJPSLegalMoves(obs, offset, &encoding);

  // std::cout << "After encode legal move." << std::endl;

  REQUIRE_EQ(offset, encoding.size());
  return encoding;
}
}  // namespace bridge_learning_env