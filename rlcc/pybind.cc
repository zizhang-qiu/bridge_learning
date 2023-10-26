//
// Created by qzz on 2023/9/23.
//
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "bridge_env.h"
#include "bridge_dataset.h"
#include "supervise_data_generator.h"
#include "playcc/pimc.h"
namespace py = pybind11;
using namespace rlcc;

PYBIND11_MODULE(bridgelearn, m
) {

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
      .def("resample", &UniformResampler::Resample);

  py::class_<SearchResult>(m, "SearchResult")
      .def_readonly("moves", &SearchResult::moves)
      .def_readonly("scores", &SearchResult::scores);

  py::class_<PIMCBot, std::shared_ptr<PIMCBot>>(m, "PIMCBot")
      .def(py::init<std::shared_ptr<Resampler>, int>())
      .def("search", &PIMCBot::Search);
}