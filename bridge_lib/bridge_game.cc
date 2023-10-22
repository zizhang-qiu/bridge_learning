#include "bridge_game.h"
#include <memory>
#include <utility>

namespace bridge_learning_env {

inline constexpr bool kIsDealerVulnerable = false;
inline constexpr bool kIsNonDealerVulnerable = false;
inline constexpr int kDefaultSeed = -1;
inline constexpr Player kDefaultDealer = kNorth;

std::shared_ptr<BridgeState> BridgeGame::NewInitialState() const {
  return std::make_shared<BridgeState>(IsDealerVulnerable(),
                                       IsNonDealerVulnerable());
}

BridgeGame::BridgeGame(const GameParameters &params) {
  params_ = params;
  is_dealer_vulnerable_ =
      ParameterValue(params_, "is_dealer_vulnerable", kIsDealerVulnerable);
  is_non_dealer_vulnerable_ = ParameterValue(
      params_, "is_non_dealer_vulnerable", kIsNonDealerVulnerable);
  dealer_ = ParameterValue(params_, "dealer", kDefaultDealer);
  seed_ = ParameterValue(params_, "seed", kDefaultSeed);
  while (seed_ == -1) {
    seed_ = static_cast<int>(std::random_device()());
  }
  rng_.seed(seed_);

  for (int uid = 0; uid < MaxMoves(); ++uid) {
    moves_.push_back(ConstructMove(uid));
  }

  for (int uid = 0; uid < MaxChanceOutcomes(); ++uid) {
    chance_outcomes_.push_back(ConstructChanceMove(uid));
  }
}

int BridgeGame::MaxMoves() const { return MaxAuctionMoves() + MaxPlayMoves(); }

int BridgeGame::GetMoveUid(BridgeMove move) const {
  return GetMoveUid(move.MoveType(), move.CardSuit(), move.CardRank(),
                    move.BidDenomination(), move.BidLevel(), move.OtherCall());
}
int BridgeGame::GetMoveUid(BridgeMove::Type move_type, Suit suit, int rank,
                           Denomination denomination, int level,
                           OtherCalls other_call) const {
  switch (move_type) {
    case BridgeMove::kPlay:return CardIndex(suit, rank);
    case BridgeMove::kAuction:
      if (other_call != kNotOtherCall) {
        return kNumCards + other_call;
      }
      return kNumCards + BidIndex(level, denomination);
    default:return -1;
  }
}
BridgeMove BridgeGame::ConstructMove(int uid) const {
  if (uid < 0 || uid > MaxMoves()) {
    return {BridgeMove::kInvalid,
        /*suit=*/kInvalidSuit,
        /*rank=*/-1, /*denomination=*/
            kInvalidDenomination,
        /*level=*/-1,
        /*other_call=*/kNotOtherCall};
  }
  if (uid < MaxPlayMoves()) {
    return {BridgeMove::kPlay,
        /*suit=*/Suit(uid % kNumSuits), /*rank=*/
            uid / kNumSuits,                /*denomination=*/
            kInvalidDenomination,
        /*level=*/-1,
        /*other_call=*/kNotOtherCall};
  }
  uid -= MaxPlayMoves();
  if (uid < kNumOtherCalls) {
    return {BridgeMove::kAuction,
        /*suit=*/kInvalidSuit,
        /*rank=*/-1, /*denomination=*/
            kInvalidDenomination,
        /*level=*/-1,
        /*other_call=*/OtherCalls(uid)};
  }
  return {BridgeMove::kAuction,
      /*suit=*/kInvalidSuit,
      /*rank=*/-1, /*denomination=*/
          Denomination((uid - kNumOtherCalls) % kNumDenominations), /*level=*/
          1 + (uid - kNumOtherCalls) / kNumDenominations,
      /*other_call=*/kNotOtherCall};
}
BridgeMove BridgeGame::ConstructChanceMove(int uid) const {
  if (uid < 0 || uid > MaxChanceOutcomes()) {
    return {BridgeMove::kInvalid,
        /*suit=*/kInvalidSuit,
        /*rank=*/-1, /*denomination=*/
            kInvalidDenomination,
        /*level=*/-1,
        /*other_call=*/kNotOtherCall};
  }
  return {BridgeMove::kDeal,
      /*suit=*/Suit(uid % kNumSuits), /*rank=*/
          uid / kNumSuits,                /*denomination=*/
          kInvalidDenomination,
      /*level=*/-1,
      /*other_call=*/kNotOtherCall};
}
bridge_learning_env::GameParameters BridgeGame::Parameters() const {
  return {{"is_dealer_vulnerable", std::to_string(IsDealerVulnerable())},
          {"is_non_dealer_vulnerable", std::to_string(IsNonDealerVulnerable())},
          {"dealer", std::to_string(dealer_)},
          {"seed", std::to_string(seed_)}};
}
BridgeMove BridgeGame::PickRandomChance(
    const std::pair<std::vector<BridgeMove>, std::vector<double>>
    &chance_outcomes) const {
  std::discrete_distribution<std::mt19937::result_type> dist(
      chance_outcomes.second.begin(), chance_outcomes.second.end());
  unsigned int index = dist(rng_);
  return chance_outcomes.first[index];
}
bool BridgeGame::IsPlayerVulnerable(Player player) const {
  return Partnership(player) == Partnership(dealer_)
         ? is_dealer_vulnerable_
         : is_non_dealer_vulnerable_;
}
bool BridgeGame::IsPartnershipVulnerable(int partnership) const {
  return partnership == Partnership(dealer_) ? is_dealer_vulnerable_
                                             : is_non_dealer_vulnerable_;
}

} // namespace bridge