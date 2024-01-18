//
// Created by qzz on 2024/1/6.
//
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "playcc/play_bot.h"
#include "playcc/pimc.h"
#include "dds_bot.h"
// #include "torch_actor.h"
// #include "torch_actor_resampler.h"
#include "playcc/alpha_mu_bot.h"

namespace py = pybind11;

PYBIND11_MODULE(bridgeplay, m) {
  py::class_<ResampleResult>(m, "ResampleResult")
      .def_readonly("success",&ResampleResult::success)
      .def_readonly("result", &ResampleResult::result);

  py::class_<Resampler, std::shared_ptr<Resampler>>(m, "Resampler");

  py::class_<UniformResampler, Resampler, std::shared_ptr<UniformResampler>>(m, "UniformResampler")
      .def(py::init<int>())
      .def("resample", &UniformResampler::Resample)
      .def("reset_with_params", &UniformResampler::ResetWithParams);

  py::class_<SearchResult>(m, "SearchResult")
      .def_readonly("moves", &SearchResult::moves)
      .def_readonly("scores", &SearchResult::scores);

  py::class_<PlayBot>(m, "PlayBot");

  py::class_<DDSBot, PlayBot>(m, "DDSBot")
      .def(py::init<>())
      .def("step", &DDSBot::Step);
  //
  py::class_<PIMCConfig>(m, "PIMCConfig")
      .def(py::init<>())
      .def_readwrite("num_worlds", &PIMCConfig::num_worlds)
      .def_readwrite("search_with_one_legal_move", &PIMCConfig::search_with_one_legal_move);

  py::class_<PIMCBot, PlayBot>(m, "PIMCBot")
      .def(py::init<std::shared_ptr<Resampler>, PIMCConfig>())
      .def("step", &PIMCBot::Step)
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
      .def("step", &VanillaAlphaMuBot::Step)
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
      .def("step", &AlphaMuBot::Step)
      .def("search", &AlphaMuBot::Search)
      .def("get_tt", &AlphaMuBot::GetTT)
      .def("set_tt", &AlphaMuBot::SetTT);

  // m.def("construct_state_from_deal",
  //       py::overload_cast<const std::array<int, ble::kNumCards> &,
  //                         const std::shared_ptr<ble::BridgeGame> &>(&ConstructStateFromDeal));

  m.def("construct_state_from_trajectory", &ConstructStateFromTrajectory);

  m.def("is_acting_player_declarer_side", &IsActingPlayerDeclarerSide);

  m.def("registered_bots", &RegisteredBots);

  m.def("is_bot_registered", &IsBotRegistered);

  m.def("load_bot",
        py::overload_cast<const std::string &,
                          const std::shared_ptr<const ble::BridgeGame> &,
                          ble::Player>(&LoadBot));
  m.def("load_bot",
        py::overload_cast<const std::string &,
                          const std::shared_ptr<const ble::BridgeGame> &,
                          ble::Player,
                          const ble::GameParameters &>(&LoadBot));
  // py::class_<TorchActor, std::shared_ptr<TorchActor>>(m, "TorchActor")
  //     .def(py::init<std::shared_ptr<rela::ModelLocker>>())
  //     .def("get_policy", &TorchActor::GetPolicy)
  //     .def("get_belief", &TorchActor::GetBelief);
  //
  // py::class_<TorchActorResampler, Resampler, std::shared_ptr<TorchActorResampler>>(m, "TorchActorResampler")
  //     .def(py::init<const std::shared_ptr<TorchActor> &,
  //                   const std::shared_ptr<ble::BridgeGame> &,
  //                   const int>())
  //     .def("resample", &TorchActorResampler::Resample);
}
