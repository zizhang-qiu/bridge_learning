//
// Created by qzz on 2023/11/3.
//
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "bridge_lib/bridge_card.h"
#include "bridge_lib/bridge_state_2.h"
#include "bridge_lib/bridge_observation.h"
#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/example_cards_ddts.h"

namespace py = pybind11;
namespace bridge_learning_env{
PYBIND11_MODULE(bridge, m){
  py::class_<Contract>(m, "Contract")
      .def("index", &Contract::Index)
      .def("__repr__", &Contract::ToString)
      .def_readwrite("level", &Contract::declarer)
      .def_readwrite("denomination", &Contract::denomination)
      .def_readwrite("double_status", &Contract::double_status)
      .def_readwrite("declarer", &Contract::declarer);

  py::class_<PlayerAction>(m, "PlayerAction")
      .def_readonly("player", &PlayerAction::player)
      .def_readonly("action", &PlayerAction::action);

  m.def("score", &Score);
  m.def("bid_index", &BidIndex);
  m.def("bid_level", &BidLevel);
  m.def("bid_denomination", &BidDenomination);
  m.def("call_string", &CallString);
  m.def("card_suit", &CardSuit);
  m.def("card_rank", &CardRank);
  m.def("card_index", &CardIndex);
  m.def("card_string", py::overload_cast<int>(&CardString));
  m.def("partnership", &Partnership);
  m.def("partner", &Partner);
  m.def("get_imp", &GetImp);

  m.def("all_contracts", &AllContracts);

  m.attr("example_deals") = example_deals;
  m.attr("example_ddts") = example_ddts;
  m.attr("NUM_PLAYERS") = kNumPlayers;
  m.attr("NUM_DENOMINATIONS") = kNumDenominations;
  m.attr("NUM_CARDS") = kNumCards;
  m.attr("NUM_BID_LEVELS") = kNumBidLevels;
  m.attr("NUM_BIDS") = kNumBids;
  m.attr("NUM_CALLS") = kNumCalls;
  m.attr("NUM_OTHER_CALLS") = kNumOtherCalls;
  m.attr("NUM_SUITS") = kNumSuits;
  m.attr("NUM_DOUBLE_STATUS") = kNumDoubleStatus;
  m.attr("NUM_PARTNERSHIPS") = kNumPartnerships;
  m.attr("NUM_CARDS_PER_SUIT") = kNumCardsPerSuit;
  m.attr("NUM_CARDS_PER_HAND") = kNumCardsPerHand;
  m.attr("NUM_CONTRACTS") = kNumContracts;
  m.attr("NUM_VULNERABILITIES") = kNumVulnerabilities;
  m.attr("NUM_TRICKS") = kNumTricks;

  py::enum_<Seat>(m, "Seat")
      .value("NORTH", Seat::kNorth)
      .value("EAST", Seat::kEast)
      .value("SOUTH", Seat::kSouth)
      .value("WEST", Seat::kWest)
      .export_values();

  py::enum_<Denomination>(m, "Denomination")
      .value("INVALID_DENOMINATION", Denomination::kInvalidDenomination)
      .value("CLUBS_TRUMP", Denomination::kClubsTrump)
      .value("DIAMONDS_TRUMP", Denomination::kDiamondsTrump)
      .value("HEARTS_TRUMP", Denomination::kHeartsTrump)
      .value("SPADES_TRUMP", Denomination::kSpadesTrump)
      .value("NO_TRUMP", Denomination::kNoTrump)
      .export_values();

  py::enum_<Suit>(m, "Suit")
      .value("INVALID_SUIT", Suit::kInvalidSuit)
      .value("CLUBS_SUIT", Suit::kClubsSuit)
      .value("DIAMONDS_SUIT", Suit::kDiamondsSuit)
      .value("HEARTS_SUIT", Suit::kHeartsSuit)
      .value("SPADES_SUIT", Suit::kSpadesSuit)
      .export_values();

  py::enum_<OtherCalls>(m, "OtherCalls")
      .value("NOT_OTHER_CALL", OtherCalls::kNotOtherCall)
      .value("PASS", OtherCalls::kPass)
      .value("DOUBLE", OtherCalls::kRedouble)
      .value("REDOUBLE", OtherCalls::kRedouble)
      .export_values();

  py::class_<BridgeCard>(m, "BridgeCard")
      .def(py::init<>())
      .def(py::init<Suit, int>())
      .def("is_valid", &BridgeCard::IsValid)
      .def("suit", &BridgeCard::CardSuit)
      .def("rank", &BridgeCard::Rank)
      .def("index", &BridgeCard::Index)
      .def("__repr__", &BridgeCard::ToString);

  py::class_<BridgeHand>(m, "BridgeHand")
      .def(py::init<>())
      .def(py::init<BridgeHand>())
      .def("cards", &BridgeHand::Cards)
      .def("add_card", &BridgeHand::AddCard)
      .def("remove_from_hand", &BridgeHand::RemoveFromHand)
      .def("is_full_hand", &BridgeHand::IsFullHand)
      .def("high_card_points", &BridgeHand::HighCardPoints)
      .def("control_value", &BridgeHand::ControlValue)
      .def("zar_high_card_points", &BridgeHand::ZarHighCardPoints)
      .def("is_card_in_hand", &BridgeHand::IsCardInHand)
      .def("__repr__", &BridgeHand::ToString);

  py::enum_<BridgeMove::Type>(m, "MoveType")
      .value("INVALID", BridgeMove::Type::kInvalid)
      .value("AUCTION", BridgeMove::Type::kAuction)
      .value("PLAY", BridgeMove::Type::kPlay)
      .value("DEAL", BridgeMove::Type::kDeal)
      .export_values();

  py::class_<BridgeMove>(m, "BridgeMove")
      .def("move_type", &BridgeMove::MoveType)
      .def("is_bid", &BridgeMove::IsBid)
      .def("bid_level", &BridgeMove::BidLevel)
      .def("bid_denomination", &BridgeMove::BidDenomination)
      .def("card_suit", &BridgeMove::CardSuit)
      .def("card_rank", &BridgeMove::CardRank)
      .def("other_call", &BridgeMove::OtherCall)
      .def("__repr__", &BridgeMove::ToString);

  py::class_<BridgeHistoryItem>(m, "BridgeHistoryItem")
      .def_readonly("move", &BridgeHistoryItem::move)
      .def_readonly("player", &BridgeHistoryItem::player)
      .def_readonly("deal_to_player", &BridgeHistoryItem::deal_to_player)
      .def_readonly("suit", &BridgeHistoryItem::suit)
      .def_readonly("rank", &BridgeHistoryItem::rank)
      .def_readonly("level", &BridgeHistoryItem::level)
      .def_readonly("denomination", &BridgeHistoryItem::denomination)
      .def_readonly("other_call", &BridgeHistoryItem::other_call)
      .def("__repr__", &BridgeHistoryItem::ToString);

  py::class_<BridgeGame, std::shared_ptr<BridgeGame>>(m, "BridgeGame")
      .def(py::init<const GameParameters>())
      .def("num_distinct_actions", &BridgeGame::NumDistinctActions)
      .def("max_chance_outcomes", &BridgeGame::MaxChanceOutcomes)
      .def("max_moves", &BridgeGame::MaxMoves)
      .def("max_utility", &BridgeGame::MaxUtility)
      .def("min_utility", &BridgeGame::MinUtility)
      .def("max_game_length", &BridgeGame::MaxGameLength)
      .def("min_game_length", &BridgeGame::MinGameLength)
      .def("parameters", &BridgeGame::Parameters)
      .def("is_dealer_vulnerable", &BridgeGame::IsDealerVulnerable)
      .def("is_non_dealer_vulnerable", &BridgeGame::IsNonDealerVulnerable)
      .def("is_player_vulnerable", &BridgeGame::IsPlayerVulnerable)
      .def("is_partnership_vulnerable", &BridgeGame::IsPartnershipVulnerable)
      .def("dealer", &BridgeGame::Dealer)
      .def("get_move", &BridgeGame::GetMove)
      .def("get_move_uid", py::overload_cast<BridgeMove>(&BridgeGame::GetMoveUid, py::const_))
      .def("get_chance_outcome", &BridgeGame::GetChanceOutcome)
      .def("pick_random_chance", &BridgeGame::PickRandomChance);

  py::enum_<Phase>(m, "Phase")
      .value("DEAL", Phase::kDeal)
      .value("AUCTION", Phase::kAuction)
      .value("PLAY", Phase::kPlay)
      .value("GAME_OVER", Phase::kGameOver)
      .export_values();

  py::class_<BridgeState2>(m, "BridgeState2")
      .def(py::init<std::shared_ptr<BridgeGame>>())
      .def("hands", &BridgeState2::Hands)
      .def("history", &BridgeState2::History)
      .def("current_player", &BridgeState2::CurrentPlayer)
      .def("apply_move", &BridgeState2::ApplyMove)
      .def("move_is_legal", &BridgeState2::MoveIsLegal)
      .def("is_terminal", &BridgeState2::IsTerminal)
      .def("parent_game", &BridgeState2::ParentGame)
      .def("chance_outcome_prob", &BridgeState2::ChanceOutcomeProb)
      .def("apply_random_chance", &BridgeState2::ApplyRandomChance)
      .def("current_phase", &BridgeState2::CurrentPhase)
      .def("legal_moves", py::overload_cast<>(&BridgeState2::LegalMoves, py::const_))
      .def("legal_moves", py::overload_cast<Player>(&BridgeState2::LegalMoves, py::const_))
      .def("score_for_contracts", &BridgeState2::ScoreForContracts)
      .def("clone", &BridgeState2::Clone)
      .def("double_dummy_results", &BridgeState2::DoubleDummyResults, py::arg("dds_order") = false)
      .def("uid_history", &BridgeState2::UidHistory)
      .def("is_chance_node", &BridgeState2::IsChanceNode)
      .def("num_cards_played", &BridgeState2::NumCardsPlayed)
      .def("scores", &BridgeState2::Scores)
      .def("get_contract", &BridgeState2::GetContract)
      .def("__repr__", &BridgeState2::ToString);

  py::class_<BridgeObservation>(m, "BridgeObservation")
      .def(py::init<const BridgeState2, Player>())
      .def("cur_player_offset", &BridgeObservation::CurPlayerOffset)
      .def("auction_history", &BridgeObservation::AuctionHistory)
      .def("hands", &BridgeObservation::Hands)
      .def("parent_game", &BridgeObservation::ParentGame)
      .def("legal_moves", &BridgeObservation::LegalMoves)
      .def("is_player_vulnerable", &BridgeObservation::IsPlayerVulnerable)
      .def("is_opponent_vulnerable", &BridgeObservation::IsOpponentVulnerable);

  py::class_<ObservationEncoder>(m, "ObservationEncoder");

  py::enum_<ObservationEncoder::Type>(m, "EncoderType")
      .value("CANONICAL", ObservationEncoder::Type::kCanonical);

  py::class_<CanonicalEncoder, ObservationEncoder>(m, "CanonicalEncoder")
      .def(py::init<std::shared_ptr<BridgeGame>>())
      .def("shape", &CanonicalEncoder::Shape)
      .def("encode", &CanonicalEncoder::Encode)
      .def("type", &CanonicalEncoder::type);
}
}

