//
// Created by qzz on 2023/11/3.
//
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "bridge_lib/bridge_card.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_observation.h"
#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/example_cards_ddts.h"

namespace py = pybind11;

void CheckPyTupleSize(const py::tuple& t, const size_t size) {
  if (t.size() != size) {
    std::cerr << "The tuple needs " << size << " items, but got " << t.size() <<
        " items!" << std::endl;
    std::abort();
  }
}

namespace bridge_learning_env {
PYBIND11_MODULE(bridge, m) {
  py::enum_<DoubleStatus>(m, "DoubleStatus")
      .value("UNDOUBLED", DoubleStatus::kUndoubled)
      .value("DOUBLED", DoubleStatus::kDoubled)
      .value("REDOUBLED", DoubleStatus::kRedoubled)
      .export_values();

  py::class_<Contract>(m, "Contract")
      .def(py::init<>())
      .def("index", &Contract::Index)
      .def("__repr__", &Contract::ToString)
      .def_readwrite("level", &Contract::level)
      .def_readwrite("denomination", &Contract::denomination)
      .def_readwrite("double_status", &Contract::double_status)
      .def_readwrite("declarer", &Contract::declarer);

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
      .def("__eq__", &BridgeCard::operator==)
      .def("is_valid", &BridgeCard::IsValid)
      .def("suit", &BridgeCard::CardSuit)
      .def("rank", &BridgeCard::Rank)
      .def("index", &BridgeCard::Index)
      .def("__repr__", &BridgeCard::ToString)
      .def(py::pickle(
          [](const BridgeCard& card) {
            //__getstate__
            return py::make_tuple(card.CardSuit(), card.Rank());
          },
          [](const py::tuple& t) {
            //__setstate__
            CheckPyTupleSize(t, 2);
            const BridgeCard card{t[0].cast<Suit>(), t[1].cast<int>()};
            return card;
          }
          ));

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
      .def("__repr__", &BridgeHand::ToString)
      .def(py::pickle(
          [](const BridgeHand& hand) {
            //__getstate__
            return py::make_tuple(hand.Cards());
          },
          [](const py::tuple& t) {
            //__setstate__
            CheckPyTupleSize(t, 1);
            BridgeHand hand{};
            const auto cards = t[0].cast<std::vector<BridgeCard>>();
            for (const auto card : cards) {
              hand.AddCard(card);
            }
            return hand;
          }));

  py::enum_<BridgeMove::Type>(m, "MoveType")
      .value("INVALID", BridgeMove::Type::kInvalid)
      .value("AUCTION", BridgeMove::Type::kAuction)
      .value("PLAY", BridgeMove::Type::kPlay)
      .value("DEAL", BridgeMove::Type::kDeal)
      .export_values();

  py::class_<BridgeMove>(m, "BridgeMove")
      .def(py::init<>())
      .def(py::init<
        const BridgeMove::Type, // move type
        const Suit,             // suit
        const int,              // rank
        const Denomination,     //denomination
        const int,              // bid level
        const OtherCalls>())
      .def("__eq__", &BridgeMove::operator==)
      .def("move_type", &BridgeMove::MoveType)
      .def("is_bid", &BridgeMove::IsBid)
      .def("bid_level", &BridgeMove::BidLevel)
      .def("bid_denomination", &BridgeMove::BidDenomination)
      .def("card_suit", &BridgeMove::CardSuit)
      .def("card_rank", &BridgeMove::CardRank)
      .def("other_call", &BridgeMove::OtherCall)
      .def("__repr__", &BridgeMove::ToString)
      .def(py::pickle(
          [](const BridgeMove& move) {
            //__getstate__
            // Use same order as constructor.
            return py::make_tuple(move.MoveType(), move.CardSuit(),
                                  move.CardRank(), move.BidDenomination(),
                                  move.BidLevel(), move.OtherCall());
          },
          [](const py::tuple& t) {
            //__setstate__
            CheckPyTupleSize(t, 6);
            const auto move_type = t[0].cast<BridgeMove::Type>();
            const Suit suit = t[1].cast<Suit>();
            const int rank = t[2].cast<int>();
            const auto denomination = t[3].cast<Denomination>();
            const int level = t[4].cast<int>();
            const auto other_call = t[5].cast<OtherCalls>();
            const BridgeMove move{
                /*move_type=*/move_type,
                              /*suit=*/suit,
                              /*rank=*/rank,
                              /*denomination=*/denomination,
                              /*level=*/level,
                              /*other_call=*/other_call};
            return move;
          }));

  py::class_<BridgeHistoryItem>(m, "BridgeHistoryItem")
      .def_readonly("move", &BridgeHistoryItem::move)
      .def_readonly("player", &BridgeHistoryItem::player)
      .def_readonly("deal_to_player", &BridgeHistoryItem::deal_to_player)
      .def_readonly("suit", &BridgeHistoryItem::suit)
      .def_readonly("rank", &BridgeHistoryItem::rank)
      .def_readonly("level", &BridgeHistoryItem::level)
      .def_readonly("denomination", &BridgeHistoryItem::denomination)
      .def_readonly("other_call", &BridgeHistoryItem::other_call)
      .def("__repr__", &BridgeHistoryItem::ToString)
      .def(py::pickle([](const BridgeHistoryItem& item) {
                        //__getstate__
                        return py::make_tuple(item.move,
                                              item.player,
                                              item.deal_to_player,
                                              item.suit,
                                              item.rank,
                                              item.denomination,
                                              item.level,
                                              item.other_call);
                      },
                      [](const py::tuple& t) {
                        //__setstate__
                        CheckPyTupleSize(t, 8);
                        const auto move = t[0].cast<BridgeMove>();
                        const auto player = t[1].cast<Player>();
                        const auto deal_to_player = t[2].cast<Player>();
                        const auto suit = t[3].cast<Suit>();
                        const int rank = t[4].cast<int>();
                        const auto denomination = t[5].cast<Denomination>();
                        const int level = t[6].cast<int>();
                        const auto other_call = t[7].cast<OtherCalls>();
                        BridgeHistoryItem item{move};
                        item.player = player;
                        item.deal_to_player = deal_to_player;
                        item.suit = suit;
                        item.rank = rank;
                        item.denomination = denomination;
                        item.level = level;
                        item.other_call = other_call;
                        return item;
                      }));

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
      .def("get_move_uid",
           py::overload_cast<BridgeMove>(&BridgeGame::GetMoveUid, py::const_))
      .def("get_chance_outcome", &BridgeGame::GetChanceOutcome)
      .def("pick_random_chance", &BridgeGame::PickRandomChance)
      .def("__repr__", &BridgeGame::ToString)
      .def(py::pickle([](const BridgeGame& game) {
                        //__getstate__
                        return py::make_tuple(game.IsDealerVulnerable(),
                                              game.IsNonDealerVulnerable(),
                                              game.Dealer(),
                                              game.Seed());
                      },
                      [](const py::tuple& t) {
                        //__setstate__
                        CheckPyTupleSize(t, 4);
                        const bool is_dealer_vulnerable = t[0].cast<bool>();
                        const bool is_non_dealer_vulnerable = t[1].cast<bool>();
                        const auto dealer = t[2].cast<Player>();
                        const int seed = t[3].cast<int>();
                        const GameParameters parameters{
                            {"is_dealer_vulnerable",
                             std::to_string(is_dealer_vulnerable)},
                            {"is_non_dealer_vulnerable",
                             std::to_string(is_non_dealer_vulnerable)},
                            {"dealer", std::to_string(dealer)},
                            {"seed", std::to_string(seed)}
                        };
                        const BridgeGame game{parameters};
                        return game;
                      }
          ));

  m.attr("default_game") = default_game;

  py::enum_<Phase>(m, "Phase")
      .value("DEAL", Phase::kDeal)
      .value("AUCTION", Phase::kAuction)
      .value("PLAY", Phase::kPlay)
      .value("GAME_OVER", Phase::kGameOver)
      .export_values();

  py::class_<BridgeState>(m, "BridgeState")
      .def(py::init<std::shared_ptr<BridgeGame>>())
      .def("__eq__", &BridgeState::operator==)
      .def("hands", &BridgeState::Hands)
      .def("history", &BridgeState::History)
      .def("current_player", &BridgeState::CurrentPlayer)
      .def("apply_move", &BridgeState::ApplyMove)
      .def("move_is_legal", &BridgeState::MoveIsLegal)
      .def("is_terminal", &BridgeState::IsTerminal)
      .def("parent_game", &BridgeState::ParentGame)
      .def("chance_outcome_prob", &BridgeState::ChanceOutcomeProb)
      .def("apply_random_chance", &BridgeState::ApplyRandomChance)
      .def("current_phase", &BridgeState::CurrentPhase)
      .def("legal_moves",
           py::overload_cast<>(&BridgeState::LegalMoves, py::const_))
      .def("legal_moves",
           py::overload_cast<Player>(&BridgeState::LegalMoves, py::const_))
      .def("score_for_contracts", &BridgeState::ScoreForContracts)
      .def("clone", &BridgeState::Clone)
      .def("double_dummy_results", &BridgeState::DoubleDummyResults,
           py::arg("dds_order") = false)
      .def("uid_history", &BridgeState::UidHistory)
      .def("is_chance_node", &BridgeState::IsChanceNode)
      .def("num_cards_played", &BridgeState::NumCardsPlayed)
      .def("scores", &BridgeState::Scores)
      .def("get_contract", &BridgeState::GetContract)
      .def("deal_history", &BridgeState::DealHistory)
      .def("auction_history", &BridgeState::AuctionHistory)
      .def("play_history", &BridgeState::PlayHistory)
      .def("__repr__", &BridgeState::ToString)
      .def(py::pickle([](const BridgeState& state) {
                        //__setstate__
                        //For a BridgeState, we need track parent game and action history.
                        return py::make_tuple(state.ParentGame().get(),
                                              state.History());
                      }, [](const py::tuple& t) {
                        //__getstate__
                        CheckPyTupleSize(t, 2);
                        const auto game = std::make_shared<BridgeGame>(
                            t[0].cast<BridgeGame>());
                        const auto move_history = t[1].cast<std::vector<
                          BridgeHistoryItem>>();
                        BridgeState state{game};
                        for (const auto& item : move_history) {
                          state.ApplyMove(item.move);
                        }
                        return state;
                      }));

  py::class_<BridgeObservation>(m, "BridgeObservation")
      .def(py::init<const BridgeState&, Player>())
      .def(py::init<const BridgeState&>())
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

  m.attr("AUCTION_TENSOR_SIZE") = kAuctionTensorSize;
  py::class_<CanonicalEncoder, ObservationEncoder>(m, "CanonicalEncoder")
      .def(py::init<std::shared_ptr<BridgeGame>, int>())
      .def("shape", &CanonicalEncoder::Shape)
      .def("encode", &CanonicalEncoder::Encode)
      .def("get_play_tensor_size", &CanonicalEncoder::GetPlayTensorSize)
      .def("get_auction_tensor_size", &CanonicalEncoder::GetAuctionTensorSize)
      .def("type", &CanonicalEncoder::type);
}
}
