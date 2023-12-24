//
// Created by qzz on 2023/9/23.
//
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "bridge_env.h"
#include "bridge_dataset.h"
#include "supervise_data_generator.h"
#include "playcc/play_bot.h"
#include "playcc/pimc.h"
#include "playcc/cheat_bot.h"
#include "playcc/alpha_mu_bot.h"

namespace py = pybind11;
using namespace rlcc;

PYBIND11_MODULE(bridgelearn, m) {

  py::class_<BridgeData>(m, "BridgeData")
      .def(py::init<>())
      .def_readwrite("deal", &BridgeData::deal)
      .def_readwrite("ddt", &BridgeData::ddt);

  py::class_<BridgeDataset, std::shared_ptr<BridgeDataset>>(m, "BridgeDataset")
      .def(py::init<std::vector<std::array<int, ble::kNumCards>>>())
      .def(py::init<std::vector<std::array<int, ble::kNumCards>>,
                    std::vector<std::array<int, kDoubleDummyResultSize>>>())
      .def("size", &BridgeDataset::Size)
      .def("next", &BridgeDataset::Next);

  py::class_<BridgeEnv, std::shared_ptr<BridgeEnv>>(m, "BridgeEnv")
      .def(py::init<ble::GameParameters, bool>())
      .def("feature_size", &BridgeEnv::FeatureSize)
      .def("reset_with_deck", &BridgeEnv::ResetWithDeck)
      .def("reset_with_deck_and_double_dummy_results", &BridgeEnv::ResetWithDeckAndDoubleDummyResults)
      .def("reset", &BridgeEnv::Reset)
      .def("set_bridge_dataset", &BridgeEnv::SetBridgeDataset)
      .def("reset_with_bridge_data", &BridgeEnv::ResetWithBridgeData)
      .def("step",
           py::overload_cast<ble::BridgeMove>(&BridgeEnv::Step)
      )
      .def("step",
           py::overload_cast<int>(&BridgeEnv::Step)
      )
      .def("terminated", &BridgeEnv::Terminated)
      .def("returns", &BridgeEnv::Returns)
      .def("current_player", &BridgeEnv::CurrentPlayer)
      .def("ble_state", &BridgeEnv::BleState)
      .def("ble_game", &BridgeEnv::BleGame)
      .def("ble_observation", &BridgeEnv::BleObservation)
      .def("get_move", &BridgeEnv::GetMove)
      .def("last_active_player", &BridgeEnv::LastActivePlayer)
      .def("last_move", &BridgeEnv::LastMove)
      .def("feature", &BridgeEnv::Feature)
      .def("parameters", &BridgeEnv::Parameters)
      .def("__repr__", &BridgeEnv::ToString);

  py::class_<BridgeVecEnv>(m, "BridgeVecEnv")
      .def(py::init<>())
      .def("append", &BridgeVecEnv::Append, py::keep_alive<1, 2>())
      .def("size", &BridgeVecEnv::Size)
      .def("display", &BridgeVecEnv::DisPlay)
      .def("step", &BridgeVecEnv::Step)
      .def("feature", &BridgeVecEnv::Feature)
      .def("reset", &BridgeVecEnv::Reset)
      .def("all_terminated", &BridgeVecEnv::AllTerminated)
      .def("any_terminated", &BridgeVecEnv::AnyTerminated);

  py::class_<SuperviseDataGenerator>(m, "SuperviseDataGenerator")
      .def(py::init<std::vector<std::vector<int>>,
                    int,
                    std::shared_ptr<ble::BridgeGame>,
                    int>())
      .def("next_batch", &SuperviseDataGenerator::NextBatch)
      .def("all_data", &SuperviseDataGenerator::AllData);

  py::class_<Resampler, std::shared_ptr<Resampler>>(m, "Resampler");

  py::class_<UniformResampler, Resampler, std::shared_ptr<UniformResampler>>(m, "UniformResampler")
      .def(py::init<int>())
      .def("resample", &UniformResampler::Resample)
      .def("reset_with_params", &UniformResampler::ResetWithParams);

  py::class_<SearchResult>(m, "SearchResult")
      .def_readonly("moves", &SearchResult::moves)
      .def_readonly("scores", &SearchResult::scores);

  py::class_<PlayBot>(m, "PlayBot");

  py::class_<CheatBot, PlayBot>(m, "CheatBot")
      .def(py::init<>())
      .def("act", &CheatBot::Act);
  //
  py::class_<PIMCConfig>(m, "PIMCConfig")
      .def(py::init<>())
      .def_readwrite("num_worlds", &PIMCConfig::num_worlds)
      .def_readwrite("search_with_one_legal_move", &PIMCConfig::search_with_one_legal_move);

  py::class_<PIMCBot, PlayBot>(m, "PIMCBot")
      .def(py::init<std::shared_ptr<Resampler>, PIMCConfig>())
      .def("act", &PIMCBot::Act)
      .def("search", &PIMCBot::Search);

  py::class_<OutcomeVector>(m, "OutcomeVector")
      .def_readwrite("game_status", &OutcomeVector::game_status)
      .def_readwrite("possible_world", &OutcomeVector::possible_world)
      .def_readwrite("move", &OutcomeVector::move)
      .def("__repr__", &OutcomeVector::ToString)
      .def("score", &OutcomeVector::Score);

  py::class_<ParetoFront>(m, "ParetoFront")
      .def(py::init<>())
      .def(py::init<std::vector<OutcomeVector>>())
      .def("size", &ParetoFront::Size)
      .def("insert", &ParetoFront::Insert)
      .def("__repr__", &ParetoFront::ToString)
      .def("outcome_vectors", &ParetoFront::OutcomeVectors)
      .def("empty", &ParetoFront::Empty)
      .def("score", &ParetoFront::Score)
      .def("best_outcome", &ParetoFront::BestOutcome)
      .def("set_move", &ParetoFront::SetMove)
      .def("pareto_front_with_one_outcome_vector", &ParetoFront::ParetoFrontWithOneOutcomeVector)
      .def("serialize", &ParetoFront::Serialize)
      .def("deserialize", &ParetoFront::Deserialize);

  m.def("pareto_front_min", &ParetoFrontMin);
  m.def("pareto_front_max", &ParetoFrontMax);
  m.def("pareto_front_dominate", &ParetoFrontDominate);

  py::class_<ble::BridgeStateWithoutHiddenInfo>(m, "BridgeStateWithoutHiddenInfo")
      .def(py::init<ble::BridgeState>())
      .def("__repr__", &ble::BridgeStateWithoutHiddenInfo::ToString)
      .def("uid_history", &ble::BridgeStateWithoutHiddenInfo::UidHistory)
      .def("apply_move", &ble::BridgeStateWithoutHiddenInfo::ApplyMove)
      .def("legal_moves", py::overload_cast<>(&ble::BridgeStateWithoutHiddenInfo::LegalMoves, py::const_))
      .def("current_player", &ble::BridgeStateWithoutHiddenInfo::CurrentPlayer)
      .def("is_terminal", &ble::BridgeStateWithoutHiddenInfo::IsTerminal)
      .def("num_declarer_tricks", &ble::BridgeStateWithoutHiddenInfo::NumDeclarerTricks)
      .def("get_contract", &ble::BridgeStateWithoutHiddenInfo::GetContract)
      .def("serialize", &ble::BridgeStateWithoutHiddenInfo::Serialize)
      .def("deserialize", &ble::BridgeStateWithoutHiddenInfo::Deserialize);

  py::class_<AlphaMuConfig>(m, "AlphaMuConfig")
      .def(py::init<>())
      .def_readwrite("num_max_moves", &AlphaMuConfig::num_max_moves)
      .def_readwrite("num_worlds", &AlphaMuConfig::num_worlds)
      .def_readwrite("search_with_one_legal_move", &AlphaMuConfig::search_with_one_legal_move)
      .def_readwrite("root_cut", &AlphaMuConfig::root_cut)
      .def_readwrite("early_cut", &AlphaMuConfig::early_cut);

  py::class_<VanillaAlphaMuBot, PlayBot>(m, "VanillaAlphaMuBot")
      .def(py::init<std::shared_ptr<Resampler>, AlphaMuConfig>())
      .def("act", &VanillaAlphaMuBot::Act)
      .def("search", &VanillaAlphaMuBot::Search);

  py::class_<TranspositionTable>(m, "TranspositionTable")
      .def(py::init<>())
      .def("table", &TranspositionTable::Table)
      .def("__item__", &TranspositionTable::operator[])
      .def("__repr__", &TranspositionTable::ToString)
      .def("serialize", &TranspositionTable::Serialize)
      .def("deserialize", &TranspositionTable::Deserialize);

  py::class_<AlphaMuBot, PlayBot>(m, "AlphaMuBot")
      .def(py::init<std::shared_ptr<Resampler>, AlphaMuConfig>())
      .def("act", &AlphaMuBot::Act)
      .def("search", &AlphaMuBot::Search)
      .def("get_tt", &AlphaMuBot::GetTT)
      .def("set_tt", &AlphaMuBot::SetTT);

  m.def("construct_state_from_deal", py::overload_cast<const std::array<int, ble::kNumCards> &,
                                                       const std::shared_ptr<ble::BridgeGame> &>(&ConstructStateFromDeal));

  m.def("construct_state_from_trajectory", &ConstructStateFromTrajectory);

  m.def("is_acting_player_declarer_side", &IsActingPlayerDeclarerSide);
}