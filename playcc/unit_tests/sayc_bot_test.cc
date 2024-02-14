//
// Created by qzz on 24-2-2.
//

#include "playcc/sayc/sayc_bot.h"

#include <playcc/utils.h>
#include <playcc/sayc/utils.h>
using namespace sayc;
const auto sayc_bot = std::make_shared<SAYCBot>();

struct TestCase {
  // Our hand.
  std::vector<std::string> card_strings;
  // Bid sequence.
  std::vector<std::string> auction_strings;

  ble::BridgeMove expected_move;
};

std::mt19937 rng;

ble::BridgeMove GetMoveFromTestCase(const TestCase& test_case,
                                    const shared_ptr<SAYCBot>& sayc_bot) {
  // Which player does the cards belong to?
  const int player = test_case.auction_strings.size() % ble::kNumPlayers;
  std::array<std::vector<std::string>, ble::kNumPlayers> cards{};
  cards[player] = test_case.card_strings;
  auto state = ConstructStateFromCardStrings(cards, ble::default_game, rng);
  for (const auto& auction_string : test_case.auction_strings) {
    const auto move = ConstructAuctionMoveFromString(auction_string);
    state.ApplyMove(move);
  }
  const ble::BridgeMove move = sayc_bot->Step({state});
  return move;
}

void NoTrumpOpeningTest() {
  sayc_bot->Restart();
  TestCase one_nt_case{
      {"SA", "S6", "S5", "HK", "HQ", "H2", "D5", "D3", "CA", "CQ", "CJ", "C9",
       "C7"},
      {},
      ConstructAuctionMoveFromString("1NT")};
  auto one_nt_state = ConstructStateFromCardStrings(
      {{one_nt_case.card_strings, {}, {}, {}}}, ble::default_game, rng);
  ble::BridgeMove one_nt_move = sayc_bot->Step({one_nt_state});
  SPIEL_CHECK_EQ(one_nt_move, one_nt_case.expected_move);

  sayc_bot->Restart();
  TestCase two_nt_case{
      {
          "SA", "SK", "S5", "HK", "HQ", "HJ", "D5", "D3", "CA", "CQ", "CJ",
          "C9", "C7"},
      {},
      ConstructAuctionMoveFromString("2NT")};
  auto two_nt_state = ConstructStateFromCardStrings(
      {{two_nt_case.card_strings, {}, {}, {}}}, ble::default_game, rng);
  ble::BridgeMove two_nt_move = sayc_bot->Step({two_nt_state});
  SPIEL_CHECK_EQ(two_nt_move, two_nt_case.expected_move);

  sayc_bot->Restart();
  TestCase three_nt_case{
      {
          "SA", "SK", "S5", "HK", "HQ", "HJ", "DK", "DQ", "CA", "CQ", "CJ",
          "C9",
          "C7"},
      {},
      ConstructAuctionMoveFromString("3NT")};
  auto three_nt_state = ConstructStateFromCardStrings(
      {{two_nt_case.card_strings, {}, {}, {}}}, ble::default_game, rng);
  ble::BridgeMove three_nt_move = sayc_bot->Step({three_nt_state});
  SPIEL_CHECK_EQ(three_nt_move, two_nt_case.expected_move);
}

void OneLevelOpeningTest() {
  // 1C
  sayc_bot->Restart();
  const TestCase one_clubs_case{
      {"SK", "SQ", "S5", "S4", "HA", "H8", "H7", "H3", "D6", "CK", "CT", "C6",
       "C4"},
      {},
      ConstructAuctionMoveFromString("1C")};
  const auto one_clubs_move = GetMoveFromTestCase(one_clubs_case, sayc_bot);
  SPIEL_CHECK_EQ(one_clubs_move, one_clubs_case.expected_move);

  // 1C special case, 3-3 minor
  sayc_bot->Restart();
  const TestCase one_clubs_special_case{
      {"SK", "S9", "S8", "HQ", "HT", "H7", "H6", "DA", "DK", "DQ", "C7", "C5",
       "C2"},
      {}, ConstructAuctionMoveFromString("1C")};
  const auto one_clubs_special_move = GetMoveFromTestCase(
      one_clubs_special_case, sayc_bot);
  SPIEL_CHECK_EQ(one_clubs_special_move, one_clubs_special_case.expected_move);

  // 1D
  sayc_bot->Restart();
  const TestCase one_diamonds_case{
      {"S9", "S8", "HQ", "H7", "H6", "DA", "DK", "DQ", "D9", "D7", "D6", "D5",
       "C7"},
      {}, ConstructAuctionMoveFromString("1D")};
  const auto one_diamonds_move = GetMoveFromTestCase(
      one_diamonds_case, sayc_bot);
  SPIEL_CHECK_EQ(one_diamonds_move, one_diamonds_case.expected_move);

  // 1D special case, 4-4 in minor
  sayc_bot->Restart();
  const TestCase one_diamonds_special_case{
      {"SK", "SQ", "S8", "S7", "HA", "DJ", "DT", "D9", "D2", "CQ", "CT", "C8",
       "C3"},
      {}, ConstructAuctionMoveFromString("1D")};
  const auto one_diamonds_special_move = GetMoveFromTestCase(
      one_diamonds_special_case, sayc_bot);
  SPIEL_CHECK_EQ(one_diamonds_special_move,
                 one_diamonds_special_case.expected_move);

  // 1S
  sayc_bot->Restart();
  const TestCase one_spades_case{
      {"SQ", "SJ", "S8", "S5", "S4", "HA", "HK", "H7", "H3", "H2", "D6", "C6",
       "C4"},
      {}, ConstructAuctionMoveFromString("1S")};
  const auto one_spades_move = GetMoveFromTestCase(
      one_spades_case, sayc_bot);
  SPIEL_CHECK_EQ(one_spades_move, one_spades_case.expected_move);

  // Fourth seat cases.
  sayc_bot->Restart();
  const TestCase fourth_seat_case1{
      {"SA", "SQ", "SJ", "S7", "S5", "HK", "H8", "H3", "D7", "D6", "D5", "C4",
       "C3"},
      {"P", "P", "P"},
      ConstructAuctionMoveFromString("1S")};
  const auto fourth_seat_case1_move = GetMoveFromTestCase(
      fourth_seat_case1, sayc_bot);
  SPIEL_CHECK_EQ(fourth_seat_case1_move, fourth_seat_case1.expected_move);

  sayc_bot->Restart();
  const TestCase fourth_seat_case2{
      {"SA", "ST", "S9", "HK", "HJ", "H3", "DJ", "D9", "D7", "D2", "CQ", "C9",
       "C8"},
      {"P", "P", "P"},
      ConstructAuctionMoveFromString("Pass")};
  const auto fourth_seat_case2_move = GetMoveFromTestCase(
      fourth_seat_case2, sayc_bot);
  SPIEL_CHECK_EQ(fourth_seat_case2_move, fourth_seat_case2.expected_move);

  sayc_bot->Restart();
  const TestCase fourth_seat_case3{
      {"S7", "S4", "S2", "HT", "H8", "DK", "D7", "D5", "CA", "CK", "CT",
       "C7", "C5"},
      {"P", "P", "P"},
      ConstructAuctionMoveFromString("Pass")};
  const auto fourth_seat_case3_move = GetMoveFromTestCase(
      fourth_seat_case3, sayc_bot);
  SPIEL_CHECK_EQ(fourth_seat_case3_move, fourth_seat_case3.expected_move);

}

int main(int argc, char* argv[]) {
  NoTrumpOpeningTest();
  OneLevelOpeningTest();
}