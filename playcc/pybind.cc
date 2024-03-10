//
// Created by qzz on 2024/1/6.
//
#include "bridge_lib/bridge_utils.h"
#include "playcc/utils.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/detail/descr.h"
#include "pybind11/operators.h"

#include "playcc/play_bot.h"
#include "playcc/pimc.h"
#include "dds_bot.h"
// #include "torch_actor.h"
// #include "torch_actor_resampler.h"
#include "deal_analyzer.h"
#include "opening_lead_evaluation_thread_loop.h"
#include "torch_actor.h"
#include "nn_belief_resampler.h"
#include "belief_based_opening_lead_bot.h"
#include "wbridge5_trajectory_bot.h"
#include "playcc/alpha_mu_bot.h"

namespace py = pybind11;

template<class BotBase=PlayBot>
class PyBot : public BotBase {
 public:
  using BotBase::BotBase;

  ~PyBot() override = default;

  ble::BridgeMove Step(const ble::BridgeState &state) override {
    PYBIND11_OVERLOAD_PURE_NAME(
        ble::BridgeMove,
        BotBase,
        "step",
        Step,
        state);
  }

  void Restart() override {
    PYBIND11_OVERLOAD_NAME(
        void,
        BotBase,
        "restart",
        Restart);
  }

  void RestartAt(const ble::BridgeState &state) override {
    PYBIND11_OVERLOAD_NAME(
        void,
        BotBase,
        "restart_at",
        RestartAt,
        state);
  }

  bool IsClonable() const override {
    PYBIND11_OVERLOAD_NAME(
        bool,
        BotBase,
        "is_clonable",
        IsClonable);
  }

  std::shared_ptr<PlayBot> Clone() override {

    PYBIND11_OVERLOAD_NAME(
        std::shared_ptr<PlayBot>,
        BotBase,
        "clone",
        Clone);
  }

  std::string Name() const override {
    PYBIND11_OVERLOAD_NAME(
        std::string,
        BotBase,
        "name",
        Name);
  }
};

template<class ResamplerBase=Resampler>
class PyResampler : public ResamplerBase {
  using ResamplerBase::ResamplerBase;

  ~PyResampler() override = default;

  ResampleResult Resample(const ble::BridgeState &state) override {
    PYBIND11_OVERLOAD_PURE_NAME(
        ResampleResult,
        ResamplerBase,
        "resample",
        Resample,
        state
    );
  }

  void ResetWithParams(const std::unordered_map<std::string, std::string> &params) override {
    PYBIND11_OVERLOAD_NAME(
        void,
        ResamplerBase,
        "reset_with_params",
        ResetWithParams,
        params
    );
  }
};

PYBIND11_MODULE(bridgeplay, m) {
  py::class_<ResampleResult>(m, "ResampleResult")
      .def(py::init<>())
      .def_readwrite("success", &ResampleResult::success)
      .def_readwrite("result", &ResampleResult::result);

  py::class_<Resampler, std::shared_ptr<Resampler>, PyResampler<Resampler>>(m, "Resampler")
      .def(py::init<>())
      .def("resample", &Resampler::Resample)
      .def("reset_with_params", &Resampler::ResetWithParams);

  py::class_<UniformResampler, Resampler, std::shared_ptr<UniformResampler>>(
      m, "UniformResampler")
      .def(py::init<int>())
      .def("resample", &UniformResampler::Resample)
      .def("reset_with_params", &UniformResampler::ResetWithParams);

  py::class_<SearchResult>(m, "SearchResult")
      .def(py::init<>())
      .def_readwrite("moves", &SearchResult::moves)
      .def_readwrite("scores", &SearchResult::scores);

  py::class_<PlayBot, PyBot<PlayBot>, std::shared_ptr<PlayBot>>(m, "PlayBot")
      .def(py::init<>())
      .def("step", &PlayBot::Step)
      .def("name", &PlayBot::Name)
      .def("is_clonable", &PlayBot::IsClonable)
      .def("clone", &PlayBot::Clone)
      .def("restart", &PlayBot::Restart)
      .def("restart_at", &PlayBot::RestartAt);

  py::class_<DDSBot, PlayBot, std::shared_ptr<DDSBot>>(m, "DDSBot")
      .def(py::init<>())
      .def("step", &DDSBot::Step);
  //
  py::class_<PIMCConfig>(m, "PIMCConfig")
      .def(py::init<>())
      .def_readwrite("num_worlds", &PIMCConfig::num_worlds)
      .def_readwrite("search_with_one_legal_move",
                     &PIMCConfig::search_with_one_legal_move);

  py::class_<PIMCBot, PlayBot, std::shared_ptr<PIMCBot>>(m, "PIMCBot")
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
      .def("pareto_front_with_one_outcome_vector",
           &ParetoFront::ParetoFrontWithOneOutcomeVector)
      .def("serialize", &ParetoFront::Serialize)
      .def("deserialize", &ParetoFront::Deserialize);

  m.def("pareto_front_min", &ParetoFrontMin);
  m.def("pareto_front_max", &ParetoFrontMax);
  m.def("pareto_front_dominate", &ParetoFrontDominate);

  py::class_<ble::BridgeStateWithoutHiddenInfo>(
      m, "BridgeStateWithoutHiddenInfo")
      .def(py::init<ble::BridgeState>())
      .def("__repr__", &ble::BridgeStateWithoutHiddenInfo::ToString)
      .def("uid_history", &ble::BridgeStateWithoutHiddenInfo::UidHistory)
      .def("apply_move", &ble::BridgeStateWithoutHiddenInfo::ApplyMove)
      .def("legal_moves",
           py::overload_cast<>(&ble::BridgeStateWithoutHiddenInfo::LegalMoves,
                               py::const_))
      .def("current_player", &ble::BridgeStateWithoutHiddenInfo::CurrentPlayer)
      .def("is_terminal", &ble::BridgeStateWithoutHiddenInfo::IsTerminal)
      .def("num_declarer_tricks",
           &ble::BridgeStateWithoutHiddenInfo::NumDeclarerTricks)
      .def("get_contract", &ble::BridgeStateWithoutHiddenInfo::GetContract)
      .def("serialize", &ble::BridgeStateWithoutHiddenInfo::Serialize)
      .def("deserialize", &ble::BridgeStateWithoutHiddenInfo::Deserialize);

  py::class_<AlphaMuConfig>(m, "AlphaMuConfig")
      .def(py::init<>())
      .def_readwrite("num_max_moves", &AlphaMuConfig::num_max_moves)
      .def_readwrite("num_worlds", &AlphaMuConfig::num_worlds)
      .def_readwrite("search_with_one_legal_move",
                     &AlphaMuConfig::search_with_one_legal_move)
      .def_readwrite("root_cut", &AlphaMuConfig::root_cut)
      .def_readwrite("early_cut", &AlphaMuConfig::early_cut)
      .def_readwrite("rollout_result", &AlphaMuConfig::rollout_result)
      .def_readwrite("verbose", &AlphaMuConfig::verbose);

  py::class_<TranspositionTable>(m, "TranspositionTable")
      .def(py::init<>())
      .def("table", &TranspositionTable::Table)
      .def("__item__", &TranspositionTable::operator[])
      .def("__repr__", &TranspositionTable::ToString)
      .def("serialize", &TranspositionTable::Serialize)
      .def("deserialize", &TranspositionTable::Deserialize);

  py::class_<AlphaMuBot, PlayBot, std::shared_ptr<AlphaMuBot>>(m, "AlphaMuBot")
      .def(py::init<std::shared_ptr<Resampler>, AlphaMuConfig>())
      .def(py::init<std::shared_ptr<Resampler>, AlphaMuConfig, ble::Player>())
      .def("step", &AlphaMuBot::Step)
      .def("restart", &AlphaMuBot::Restart)
      .def("search", &AlphaMuBot::Search)
      .def("get_tt", &AlphaMuBot::GetTT)
      .def("set_tt", &AlphaMuBot::SetTT);

  m.def("construct_state_from_deal",
        &ConstructStateFromDeal<std::vector<int>>);

  m.def("construct_state_from_deal_and_original_state", &ConstructStateFromDealAndOriginalState<std::vector<int>>);

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

  py::enum_<RolloutResult>(m, "RolloutResult")
      .value("WIN_LOSE", RolloutResult::kWinLose)
      .value("NUM_FUTURE_TRICKS", RolloutResult::kNumFutureTricks)
      .value("NUM_TOTAL_TRICKS", RolloutResult::kNumTotalTricks)
      .export_values();

  py::class_<DDSEvaluator, std::shared_ptr<DDSEvaluator>>(m, "DDSEvaluator")
      .def(py::init<>())
      .def("rollout", &DDSEvaluator::Rollout)
      .def("play_deal_to_dds_deal", &DDSEvaluator::PlayStateToDDSdeal)
      .def("auction_deal_to_dds_table_deal",
           &DDSEvaluator::AuctionStateToDDSddTableDeal)
      .def("dds_moves", &DDSEvaluator::DDSMoves);

  py::class_<TorchActor, std::shared_ptr<TorchActor>>(m, "TorchActor")
      .def(py::init<std::shared_ptr<rela::BatchRunner>>())
      .def("get_policy", &TorchActor::GetPolicy)
      .def("get_belief", &TorchActor::GetBelief);
  //
  py::class_<NNBeliefResampler, Resampler, std::shared_ptr<
      NNBeliefResampler>>(m, "NNBeliefResampler")
      .def(py::init<const std::shared_ptr<TorchActor> &, // torch actor
                    const std::shared_ptr<ble::BridgeGame> &, // game
                    const int>()) // seed
      .def("resample", &NNBeliefResampler::Resample);

  py::class_<BeliefBasedOpeningLeadBotConfig>(m, "BeliefBasedOpeningLeadBotConfig")
      .def(py::init())
      .def_readwrite("num_worlds", &BeliefBasedOpeningLeadBotConfig::num_worlds)
      .def_readwrite("num_max_sample",
                     &BeliefBasedOpeningLeadBotConfig::num_max_sample)
      .def_readwrite("fill_with_uniform_sample",
                     &BeliefBasedOpeningLeadBotConfig::fill_with_uniform_sample)
      .def_readwrite("rollout_result",
                     &BeliefBasedOpeningLeadBotConfig::rollout_result)
      .def_readwrite("verbose", &BeliefBasedOpeningLeadBotConfig::verbose);

  py::class_<NNBeliefOpeningLeadBot, PlayBot, std::shared_ptr<
      NNBeliefOpeningLeadBot>>(m, "NNBeliefOpeningLeadBot")
      .def(py::init<const std::shared_ptr<TorchActor> &,
                    const std::shared_ptr<ble::BridgeGame> &,
                    const int, // seed
                    const std::shared_ptr<DDSEvaluator> &,
                    const BeliefBasedOpeningLeadBotConfig &>(),
           py::arg("torch_actor"),
           py::arg("game"),
           py::arg("seed"),
           py::arg("evaluator"),
           py::arg("cfg"))
      .def("step", &NNBeliefOpeningLeadBot::Step);

  m.def("dds_moves", &DDSMoves);

  py::class_<ThreadedQueue<int>, std::shared_ptr<ThreadedQueue<int>>>(
      m, "ThreadedQueueInt")
      .def(py::init<int>(), py::arg("max_size"))
      .def("pop", py::overload_cast<>(&ThreadedQueue<int>::Pop))
      .def("empty", &ThreadedQueue<int>::Empty)
      .def("size", &ThreadedQueue<int>::Size);

  py::class_<OpeningLeadEvaluationThreadLoop, rela::ThreadLoop, std::shared_ptr<
      OpeningLeadEvaluationThreadLoop>>(
      m, "OpeningLeadEvaluationThreadLoop")
      .def(py::init<const std::shared_ptr<DDSEvaluator> &,
                    const std::shared_ptr<PlayBot> &, //bot
                    const std::shared_ptr<ble::BridgeGame> &, //game
                    const std::vector<std::vector<int>> &, //trajectories
                    ThreadedQueue<int> *, //queue
                    const int, //thread_idx
                    const bool>(), //verbose
           py::arg("dds_evaluator"), py::arg("bot"), py::arg("game"),
           py::arg("trajectories"), py::arg("bot_evaluation"),
           py::arg("thread_idx") = 0,
           py::arg("verbose") = false);

  py::class_<WBridge5TrajectoryBot, PlayBot, std::shared_ptr<
      WBridge5TrajectoryBot>>(m, "WBridge5TrajectoryBot")
      .def(py::init<const std::vector<std::vector<int>> &,
                    const std::shared_ptr<ble::BridgeGame> &>())
      .def("step", &WBridge5TrajectoryBot::Step);

    m.def("test", &Test);
}