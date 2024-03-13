//
// Created by qzz on 2023/9/23.
//

#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <memory>
#include <utility>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "belief_data_gen.h"
#include "bridge_actor.h"
#include "bridge_dataset.h"
#include "bridge_env.h"
#include "env.h"
#include "rlcc/bridge_env_actor.h"
#include "rlcc/duplicate_env.h"
#include "rlcc/env.h"
#include "rlcc/env_actor.h"
#include "rlcc/env_actor_thread_loop.h"
#include "supervise_data_generator.h"

namespace py = pybind11;
using namespace rlcc;

PYBIND11_MODULE(bridgelearn, m) {
  py::class_<BridgeData>(m, "BridgeData")
      .def(py::init<>())
      .def_readwrite("deal", &BridgeData::deal)
      .def_readwrite("ddt", &BridgeData::ddt);

  py::class_<BridgeDataset, std::shared_ptr<BridgeDataset>>(m, "BridgeDataset")
      .def(py::init<std::vector<std::vector<int>>>())
      .def(py::init<std::vector<std::vector<int>>,
                    std::vector<std::array<int, kDoubleDummyResultSize>>>())
      .def("size", &BridgeDataset::Size)
      .def("next", &BridgeDataset::Next);

  py::class_<BridgeEnvOptions>(m, "BridgeEnvOptions")
      .def(py::init<>())
      .def_readwrite("bidding_phase", &BridgeEnvOptions::bidding_phase)
      .def_readwrite("playing_phase", &BridgeEnvOptions::playing_phase)
      .def_readwrite("pbe_feature", &BridgeEnvOptions::pbe_feature)
      .def_readwrite("jps_feature", &BridgeEnvOptions::jps_feature)
      .def_readwrite("dnns_feature", &BridgeEnvOptions::dnns_feature)
      .def_readwrite("verbose", &BridgeEnvOptions::verbose);

  py::class_<EnvSpec>(m, "EnvSpec")
      .def_readwrite("num_players", &EnvSpec::num_players)
      .def_readwrite("num_partnerships", &EnvSpec::num_partnerships);

  py::class_<GameEnv, std::shared_ptr<GameEnv>>(m, "GameEnv");

  py::class_<BridgeEnv, GameEnv, std::shared_ptr<BridgeEnv>>(m, "BridgeEnv")
      .def(py::init<ble::GameParameters, BridgeEnvOptions>())
      .def("feature_size", &BridgeEnv::FeatureSize)
      .def("reset_with_deck", &BridgeEnv::ResetWithDeck)
      .def("reset_with_deck_and_double_dummy_results",
           &BridgeEnv::ResetWithDeckAndDoubleDummyResults)
      .def("reset", &BridgeEnv::Reset)
      .def("set_bridge_dataset", &BridgeEnv::SetBridgeDataset)
      .def("reset_with_bridge_data", &BridgeEnv::ResetWithDataSet)
      .def("step", py::overload_cast<const ble::BridgeMove&>(&BridgeEnv::Step))
      .def("step", py::overload_cast<int>(&BridgeEnv::Step))
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
      .def("spec", &BridgeEnv::Spec)
      .def("__repr__", &BridgeEnv::ToString);

  py::class_<DuplicateEnv, GameEnv, std::shared_ptr<DuplicateEnv>>(
      m, "DuplicateEnv")
      .def(py::init<const ble::GameParameters&, const BridgeEnvOptions&>())
      .def(py::init<const ble::GameParameters&, const BridgeEnvOptions&,
                    const std::shared_ptr<BridgeDataset>&>())
      .def("set_bridge_dataset", &DuplicateEnv::SetBridgeDataset)
      .def("max_num_action", &DuplicateEnv::MaxNumAction)
      .def("reset", &DuplicateEnv::Reset)
      .def("step", &DuplicateEnv::Step)
      .def("terminated", &DuplicateEnv::Terminated)
      .def("current_player", &DuplicateEnv::CurrentPlayer)
      .def("player_reward", &DuplicateEnv::PlayerReward)
      .def("rewards", &DuplicateEnv::Rewards)
      .def("__repr__", &DuplicateEnv::ToString)
      .def("game_index", &DuplicateEnv::GameIndex)
      .def("current_partnership", &DuplicateEnv::CurrentPartnership)
      .def("legal_actions", &DuplicateEnv::LegalActions)
      .def("feature", &DuplicateEnv::Feature)
      .def("spec", &DuplicateEnv::Spec);

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
      .def(py::init<std::vector<std::vector<int>>, int,
                    std::shared_ptr<ble::BridgeGame>, int>())
      .def("next_batch", &SuperviseDataGenerator::NextBatch)
      .def("all_data", &SuperviseDataGenerator::AllData);

  // py::class_<BeliefGenThreadloop, rela::ThreadLoop, std::shared_ptr<BeliefGenThreadloop>>(m, "BeliefGenThreadloop")
  //     .def(py::init<const std::shared_ptr<BeliefActor> &>());

  py::class_<BeliefDataGen, std::shared_ptr<BeliefDataGen>>(m, "BeliefDataGen")
      .def(py::init<const std::vector<std::vector<int>>&,    // trajectories
                    const int,                               //batch size
                    const std::shared_ptr<ble::BridgeGame>&  //game
                    >())
      .def("next_batch", &BeliefDataGen::NextBatch)
      .def("all_data", &BeliefDataGen::AllData);

  py::class_<Actor, std::shared_ptr<Actor>>(m, "Actor");

  py::class_<BridgeA2CActor, Actor, std::shared_ptr<BridgeA2CActor>>(
      m, "BridgeA2CActor")
      .def(py::init<const std::shared_ptr<rela::BatchRunner>&>())
      .def("observe_before_act", &BridgeA2CActor::ObserveBeforeAct)
      .def("act", &BridgeA2CActor::Act)
      .def("observe_after_act", &BridgeA2CActor::ObserveAfterAct);

  py::class_<AllPassActor, Actor, std::shared_ptr<AllPassActor>>(m,
                                                                 "AllPassActor")
      .def(py::init<>())
      .def("observe_before_act", &AllPassActor::ObserveBeforeAct)
      .def("act", &AllPassActor::Act)
      .def("observe_after_act", &AllPassActor::ObserveAfterAct);

  py::class_<JPSActor, Actor, std::shared_ptr<JPSActor>>(
      m, "BaselineActor")
      .def(py::init<const std::shared_ptr<rela::BatchRunner>&>())
      .def("observe_before_act", &JPSActor::ObserveBeforeAct)
      .def("act", &JPSActor::Act)
      .def("observe_after_act", &JPSActor::ObserveAfterAct);

  py::class_<EnvActorOptions>(m, "EnvActorOptions")
      .def(py::init<>())
      .def_readwrite("eval", &EnvActorOptions::eval);

  py::class_<EnvActor, std::shared_ptr<EnvActor>>(m, "EnvActor");

  py::class_<BridgeEnvActor, EnvActor, std::shared_ptr<BridgeEnvActor>>(
      m, "BridgeEnvActor")
      .def(py::init<const std::shared_ptr<GameEnv>&, const EnvActorOptions&,
                    std::vector<std::shared_ptr<Actor>>>(), py::keep_alive<1, 2>(), py::keep_alive<1, 4>())
      .def("observe_before_act", &BridgeEnvActor::ObserveBeforeAct)
      .def("act", &BridgeEnvActor::Act)
      .def("observe_after_act", &BridgeEnvActor::ObserveAfterAct)
      .def("send_experience", &BridgeEnvActor::SendExperience)
      .def("post_send_experience", &BridgeEnvActor::PostSendExperience)
      .def("get_env", &BridgeEnvActor::GetEnv)
      .def("history_rewards", &BridgeEnvActor::HistoryRewards)
      .def("terminal_count", &BridgeEnvActor::TerminalCount)
      .def("history_info", &BridgeEnvActor::HistoryInfo);

  py::class_<EnvActorThreadLoop, rela::ThreadLoop,
             std::shared_ptr<EnvActorThreadLoop>>(m, "EnvActorThreadLoop")
      .def(py::init<std::vector<std::shared_ptr<EnvActor>>, int, int, bool>(),
           py::arg("env_actors"), py::arg("num_game_per_env") = -1,
           py::arg("thread_idx") = -1, py::arg("verbose") = false)
        .def("main_loop", &EnvActorThreadLoop::mainLoop);
}
