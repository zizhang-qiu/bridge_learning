#ifndef BRIDGE_UTILS
#define BRIDGE_UTILS
#include <string>
#include <vector>

namespace bridge {
using Player = int;
using Action = int;

struct PlayerAction{
    Player player;
    Action action;
};

inline constexpr int kChancePlayerId = -1;
inline constexpr int kInvalidPlayer = -3;
inline constexpr int kTerminalPlayerId = -4;

inline constexpr int kNumPlayers = 4;  // N,E,S,W
inline constexpr int kNumPartnerships = 2;
inline constexpr int kNumCards = 52;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumCardsPerSuit = 13;
inline constexpr int kNumBidLevels = 7;      // 1-7
inline constexpr int kNumDenominations = 5;  // C,D,H,S,N
inline constexpr int kNumBids = kNumDenominations * kNumBidLevels;
inline constexpr int kNumOtherCalls = 3;  // Pass, Double, Redouble
inline constexpr int kNumCalls = kNumBids + kNumOtherCalls;
inline constexpr int kNumVulnerabilities = 2;
inline constexpr int kMaxUtility = 7600;
inline constexpr int kMinUtility = -7600;
inline constexpr int kMaxAuctionLength = 318;
inline constexpr int kNumCardsPerHand = kNumCards / kNumPlayers;
inline constexpr int kNumTricks = kNumCardsPerHand;


constexpr char kSuitChar[] = "CDHS";
constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kLevelChar[] = "-1234567";
constexpr char kDenominationChar[] = "CDHSN";
constexpr char kPlayerChar[] = "NESW";

enum Denomination { kClubs, kDiamonds, kHearts, kSpades, kNoTrump };
enum Seat { kNorth, kEast, kSouth, kWest };
enum class Suit { kClubs, kDiamonds, kHearts, kSpades };
enum Calls { kPass = 0, kDouble, kRedouble };
enum DoubleStatus { kUndoubled = 1, kDoubled = 2, kRedoubled = 4 };
inline constexpr int kNumDoubleStates = 3;

const std::vector<Seat> kAllSeats = {kNorth, kEast, kSouth, kWest};
const std::vector<Suit> kAllSuits = {Suit::kClubs, Suit::kDiamonds, Suit::kHearts, Suit::kSpades};
const std::vector<Suit> kAllSuitsReverse = {Suit::kSpades, Suit::kHearts, Suit::kDiamonds, Suit::kClubs};
const std::vector<DoubleStatus> kAllDoubleStatus = {kUndoubled, kDoubled, kRedoubled};

inline constexpr int kBiddingActionBase = kNumCards;
inline constexpr int kFirstBid = kRedouble + 1;

// The calls are represented in sequence: Pass, Dbl, RDbl, 1C, 1D, 1H, 1S, etc.
int Bid(int level, Denomination denomination);
int BidLevel(int bid);
Denomination BidDenomination(int bid);
std::string CallString(int call);

// Cards are represented as rank * kNumSuits + suit.
Suit CardSuit(int card);
int CardRank(int card);
int Card(Suit suit, int rank);
std::string CardString(int card);

int Partnership(Player player);
int Partner(Player player);
}  // namespace bridge

#endif /* BRIDGE_UTILS */
