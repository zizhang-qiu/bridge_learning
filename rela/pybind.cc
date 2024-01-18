//
// Created by qzz on 2023/9/19.
//
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/extension.h"

// #include "batch_runner.h"
#include "batcher.h"
// #include "../rlcc/belief_actor.h"
// #include "model_locker.h"
// #include "model.h"
#include "context.h"
// #include "future_actor.h"
#include "prioritized_replay.h"
#include "prioritized_replay2.h"
#include "thread_loop.h"
#include "transition.h"

namespace py = pybind11;
using namespace rela;

PYBIND11_MODULE(rela, m) {
  py::class_<RNNTransition, std::shared_ptr<RNNTransition>>(m, "RNNTransition")
      .def_readwrite("obs", &RNNTransition::obs)
      .def_readwrite("h0", &RNNTransition::h0)
      .def_readwrite("action", &RNNTransition::action)
      .def_readwrite("reward", &RNNTransition::reward)
      .def_readwrite("terminal", &RNNTransition::terminal)
      .def_readwrite("bootstrap", &RNNTransition::bootstrap)
      .def_readwrite("seq_len", &RNNTransition::seqLen);

  py::class_<RNNPrioritizedReplay, std::shared_ptr<RNNPrioritizedReplay>>(
        m,
        "RNNPrioritizedReplay")
      .def(py::init<
        int, // capacity,
        int, // seed,
        float, // alpha, priority exponent
        float, // beta, importance sampling exponent
        int>())
      .def("clear", &RNNPrioritizedReplay::clear)
      .def("terminate", &RNNPrioritizedReplay::terminate)
      .def("size", &RNNPrioritizedReplay::size)
      .def("num_add", &RNNPrioritizedReplay::numAdd)
      .def("sample", &RNNPrioritizedReplay::sample)
      .def("update_priority", &RNNPrioritizedReplay::updatePriority)
      .def("get", &RNNPrioritizedReplay::get);

  py::class_<TensorDictReplay, std::shared_ptr<TensorDictReplay>>(m, "TensorDictReplay")
      .def(py::init<
        int, // capacity,
        int, // seed,
        float, // alpha, priority exponent
        float, // beta, importance sampling exponent
        int>())
      .def("size", &TensorDictReplay::size)
      .def("num_add", &TensorDictReplay::numAdd)
      .def("sample", &TensorDictReplay::sample)
      .def("update_priority", &TensorDictReplay::updatePriority)
      .def("get", &TensorDictReplay::get);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  py::class_<Context>(m, "Context")
      .def(py::init<>())
      .def("push_thread_loop", &Context::pushThreadLoop, py::keep_alive<1, 2>())
      .def("start", &Context::start)
      .def("pause", &Context::pause)
      .def("resume", &Context::resume)
      .def("join", &Context::join)
      .def("terminated", &Context::terminated);

  // py::class_<ModelLocker, std::shared_ptr<ModelLocker>>(m, "ModelLocker")
  //     .def(py::init<const std::vector<py::object> &, const std::string &>())
  //     .def("update_model", &ModelLocker::updateModel);
  //
  // py::class_<BatchProcessorUnit, std::shared_ptr<BatchProcessorUnit>>(
  //       m,
  //       "BatchProcessor")
  //     .def(py::init<std::shared_ptr<ModelLocker>, const std::string &, int,
  //                   const std::string &>());
  //
  // py::class_<Models, std::shared_ptr<Models>>(m, "Models")
  //     .def(py::init<>())
  //     .def("add", &Models::add, py::keep_alive<1, 2>());

  py::class_<PrioritizedReplay2, std::shared_ptr<PrioritizedReplay2>>(
        m,
        "PrioritizedReplay2")
      .def(py::init<int, // capacity,
                    int, // seed,
                    float, // alpha, priority exponent
                    float, // beta, importance sampling exponent
                    bool, // whther we do prefetch
                    int>()) //batchdim axis (usually it is 0, if we use LSTM then this can be 1)
      .def("size", &PrioritizedReplay2::size)
      .def("num_add", &PrioritizedReplay2::numAdd)
      .def("sample", &PrioritizedReplay2::sample)
      .def("update_priority", &PrioritizedReplay2::updatePriority)
      .def("keep_priority", &PrioritizedReplay2::keepPriority);

  // py::class_<FutureActor, std::shared_ptr<FutureActor>>(m, "FutureActor");
  //
  // py::class_<BeliefActor, FutureActor, std::shared_ptr<BeliefActor>>(m, "BeliefActor")
  //     .def(py::init<const std::shared_ptr<Models> &,
  //                   const int,
  //                   const float,
  //                   const std::shared_ptr<PrioritizedReplay2> &>());

  // py::class_<BatchRunner, std::shared_ptr<BatchRunner>>(m, "BatchRunner")
  //     .def(py::init<
  //         py::object,
  //         const std::string&,
  //         int,
  //         const std::vector<std::string>&>())
  //     .def(py::init<py::object, const std::string&>())
  //     .def("add_method", &BatchRunner::addMethod)
  //     .def("start", &BatchRunner::start)
  //     .def("stop", &BatchRunner::stop)
  //     .def("update_model", &BatchRunner::updateModel)
  //     .def("block_call", &BatchRunner::blockCall)
  //     .def("set_log_freq", &BatchRunner::setLogFreq);
}
