#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_UTILS_H
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_UTILS_H
#include <string>
#include <vector>

namespace bridge_learning_env {
using Player = int;
using Action = int;

struct PlayerAction {
  Player player;
  Action action;
};

inline constexpr int kChancePlayerId = -1;
inline constexpr int kInvalidPlayer = -3;
inline constexpr int kTerminalPlayerId = -4;

inline constexpr int kNumPlayers = 4; // N,E,S,W
inline constexpr int kNumPartnerships = 2;
inline constexpr int kNumCards = 52;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumCardsPerSuit = 13;
inline constexpr int kNumBidLevels = 7;     // 1-7
inline constexpr int kNumDenominations = 5; // C,D,H,S,N
inline constexpr int kNumBids = kNumDenominations * kNumBidLevels;
inline constexpr int kNumOtherCalls = 3; // Pass, Double, Redouble
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
// constexpr char kSuitUnicode[][4] = {"\u2663", "\u2666", "\u2665", "\u2660"};

enum Denomination {
  kInvalidDenomination = -1,
  kClubsTrump,
  kDiamondsTrump,
  kHeartsTrump,
  kSpadesTrump,
  kNoTrump
};
enum Seat { kNorth, kEast, kSouth, kWest };
enum Suit {
  kInvalidSuit = -1,
  kClubsSuit,
  kDiamondsSuit,
  kHeartsSuit,
  kSpadesSuit
};

enum Vulnerability{kNorthSouth, kEastWest, kAll, kNone };
enum OtherCalls { kNotOtherCall=-1, kPass = 0, kDouble, kRedouble };
enum DoubleStatus { kUndoubled = 1, kDoubled = 2, kRedoubled = 4 };
inline constexpr int kNumDoubleStatus = 3;

const std::vector<Seat> kAllSeats = {kNorth, kEast, kSouth, kWest};
const std::vector<Suit> kAllSuits = {Suit::kClubsSuit, Suit::kDiamondsSuit,
                                     Suit::kHeartsSuit, Suit::kSpadesSuit};
const std::vector<Suit> kAllSuitsReverse = {
    Suit::kSpadesSuit, Suit::kHeartsSuit, Suit::kDiamondsSuit,
    Suit::kClubsSuit};
const std::vector<DoubleStatus> kAllDoubleStatus = {kUndoubled, kDoubled,
                                                    kRedoubled};

inline constexpr int kBiddingActionBase = kNumCards;
inline constexpr int kFirstBid = kRedouble + 1;

// The calls are represented in sequence: Pass, Dbl, RDbl, 1C, 1D, 1H, 1S, etc.
int BidIndex(int level, Denomination denomination);
int BidLevel(int bid);
Denomination BidDenomination(int bid);
std::string CallString(int call);

// Cards are represented as rank * kNumSuits + suit.
Suit CardSuit(int card);
int CardRank(int card);
int CardIndex(Suit suit, int rank);
std::string CardString(int card);
std::string CardString(Suit suit, int rank);

char SuitIndexToChar(Suit suit);
char RankIndexToChar(int rank);
char DenominationIndexToChar(Denomination denomination);
char LevelIndexToChar(int level);
std::string OtherCallsToString(OtherCalls call);

int Partnership(Player player);
int Partner(Player player);
int OpponentPartnership(int partnership);

// Double Dummy Solver encodes the suit and denomination as
// Spades=0, Heart, Diamonds, Clubs, NoTrump
// These functions converts the suit and denomination between this library and
// DDS.
Suit DDSSuitToSuit(int suit);
int SuitToDDSSuit(Suit suit);
Denomination DDSStrainToDenomination(int strain);
int DenominationToDDSStrain(Denomination denomination);
int RankToDDSRank(int rank);
int DDSRankToRank(int dds_rank);

} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_UTILS_H
