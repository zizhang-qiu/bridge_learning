//
// Created by qzz on 2023/9/23.
//

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "bridge_env.h"
#include "bridge_dataset.h"
#include "supervise_data_generator.h"
#include "belief_data_gen.h"

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
      .def_readwrite("verbose", &BridgeEnvOptions::verbose);

  py::class_<BridgeEnv, std::shared_ptr<BridgeEnv>>(m, "BridgeEnv")
      .def(py::init<ble::GameParameters, BridgeEnvOptions>())
      .def("feature_size", &BridgeEnv::FeatureSize)
      .def("reset_with_deck", &BridgeEnv::ResetWithDeck)
      .def("reset_with_deck_and_double_dummy_results", &BridgeEnv::ResetWithDeckAndDoubleDummyResults)
      .def("reset", &BridgeEnv::Reset)
      .def("set_bridge_dataset", &BridgeEnv::SetBridgeDataset)
      .def("reset_with_bridge_data", &BridgeEnv::ResetWithDataSet)
      .def("step",
           py::overload_cast<const ble::BridgeMove &>(&BridgeEnv::Step)
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

  // py::class_<BeliefGenThreadloop, rela::ThreadLoop, std::shared_ptr<BeliefGenThreadloop>>(m, "BeliefGenThreadloop")
  //     .def(py::init<const std::shared_ptr<BeliefActor> &>());

  py::class_<BeliefDataGen, std::shared_ptr<BeliefDataGen>>(m, "BeliefDataGen")
      .def(py::init<const std::vector<std::vector<int>> &, // trajectories
                    const int, //batch size
                    const std::shared_ptr<ble::BridgeGame> & //game
      >())
      .def("next_batch", &BeliefDataGen::NextBatch)
      .def("all_data", &BeliefDataGen::AllData);
}
