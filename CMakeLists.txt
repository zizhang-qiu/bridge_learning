cmake_minimum_required(VERSION 3.10.0)
project(bridge_learning VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)


add_library(double_dummy_solver OBJECT
        bridge_lib/third_party/dds/include/dll.h
        bridge_lib/third_party/dds/include/portab.h
        bridge_lib/third_party/dds/src/ABsearch.cpp
        bridge_lib/third_party/dds/src/ABsearch.h
        bridge_lib/third_party/dds/src/ABstats.cpp
        bridge_lib/third_party/dds/src/ABstats.h
        bridge_lib/third_party/dds/src/CalcTables.cpp
        bridge_lib/third_party/dds/src/CalcTables.h
        bridge_lib/third_party/dds/src/dds.cpp
        bridge_lib/third_party/dds/src/dds.h
        bridge_lib/third_party/dds/src/DealerPar.cpp
        bridge_lib/third_party/dds/src/debug.h
        bridge_lib/third_party/dds/src/dump.cpp
        bridge_lib/third_party/dds/src/dump.h
        bridge_lib/third_party/dds/src/File.cpp
        bridge_lib/third_party/dds/src/File.h
        bridge_lib/third_party/dds/src/Init.cpp
        bridge_lib/third_party/dds/src/Init.h
        bridge_lib/third_party/dds/src/LaterTricks.cpp
        bridge_lib/third_party/dds/src/LaterTricks.h
        bridge_lib/third_party/dds/src/Memory.cpp
        bridge_lib/third_party/dds/src/Memory.h
        bridge_lib/third_party/dds/src/Moves.cpp
        bridge_lib/third_party/dds/src/Moves.h
        bridge_lib/third_party/dds/src/Par.cpp
        bridge_lib/third_party/dds/src/parallel.h
        bridge_lib/third_party/dds/src/PBN.cpp
        bridge_lib/third_party/dds/src/PBN.h
        bridge_lib/third_party/dds/src/PlayAnalyser.cpp
        bridge_lib/third_party/dds/src/PlayAnalyser.h
        bridge_lib/third_party/dds/src/QuickTricks.cpp
        bridge_lib/third_party/dds/src/QuickTricks.h
        bridge_lib/third_party/dds/src/Scheduler.cpp
        bridge_lib/third_party/dds/src/Scheduler.h
        bridge_lib/third_party/dds/src/SolveBoard.cpp
        bridge_lib/third_party/dds/src/SolveBoard.h
        bridge_lib/third_party/dds/src/SolverIF.cpp
        bridge_lib/third_party/dds/src/SolverIF.h
        bridge_lib/third_party/dds/src/System.cpp
        bridge_lib/third_party/dds/src/System.h
        bridge_lib/third_party/dds/src/ThreadMgr.cpp
        bridge_lib/third_party/dds/src/ThreadMgr.h
        bridge_lib/third_party/dds/src/Timer.cpp
        bridge_lib/third_party/dds/src/Timer.h
        bridge_lib/third_party/dds/src/TimerGroup.cpp
        bridge_lib/third_party/dds/src/TimerGroup.h
        bridge_lib/third_party/dds/src/TimerList.cpp
        bridge_lib/third_party/dds/src/TimerList.h
        bridge_lib/third_party/dds/src/TimeStat.cpp
        bridge_lib/third_party/dds/src/TimeStat.h
        bridge_lib/third_party/dds/src/TimeStatList.cpp
        bridge_lib/third_party/dds/src/TimeStatList.h
        bridge_lib/third_party/dds/src/TransTable.h
        bridge_lib/third_party/dds/src/TransTableL.cpp
        bridge_lib/third_party/dds/src/TransTableL.h
        bridge_lib/third_party/dds/src/TransTableS.cpp
        bridge_lib/third_party/dds/src/TransTableS.h)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CONDA_PREFIX "D:/ProgramData/Anaconda/envs/bridge_research")
if (WIN32)
    message("Compiling on Windows")

    set(PYTHON_EXECUTABLE "${CONDA_PREFIX}/python.exe")
    set(PYTHON_INCLUDE_DIR "${CONDA_PREFIX}/include")
    set(PYTHON_LIBRARIES "${CONDA_PREFIX}/libs/python38.lib")
else ()
    message("Compiling on Linux")
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYBIND11_COMPILER_TYPE=\\\"_gcc\\\" -DPYBIND11_STDLIB=\\\"_libstdcpp\\\" -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
    set(CMAKE_CUDA_FLAGS "-std=c++17")
    SET(CUDA_HOME /usr/local/cuda-11.6)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.6")
    #    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda-11.8)
    set(CUDA_NVCC_EXECUTABLE /usr/local/cuda-11.6/bin/nvcc)
    set(CMAKE_CUDA_EXTENSIONS ON)
    set(PYTHON_EXECUTABLE /home/work/miniconda3/envs/bridge_research/bin/python)
endif ()
set(PYBIND11_FINDPYTHON ON)
#find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
#find_package(PythonLibs REQUIRED)

execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)

message(STATUS TorchPath=${TorchPath})

list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

if (WIN32)
    set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/torch_python.lib")
else ()
    set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_subdirectory(third_party/pybind11)
add_library(bridge_lib bridge_lib/bridge_utils.cc bridge_lib/bridge_game.cc bridge_lib/utils.cc bridge_lib/bridge_scoring.cc bridge_lib/bridge_state.cc bridge_lib/trick.cc
        bridge_lib/canonical_encoder.h
        bridge_lib/example_cards_ddts.h
        bridge_lib/canonical_encoder.cc
        bridge_lib/bridge_observation.cc
        bridge_lib/bridge_observation.h
        bridge_lib/bridge_card.cc
        bridge_lib/bridge_card.h
        bridge_lib/bridge_hand.cc
        bridge_lib/bridge_hand.h
        bridge_lib/bridge_move.cc
        bridge_lib/bridge_move.h
        bridge_lib/bridge_state_2.cc
        bridge_lib/bridge_state_2.h
        bridge_lib/bridge_history_item.cc
        bridge_lib/bridge_history_item.h
        bridge_lib/auction_tracker.cc
        bridge_lib/auction_tracker.h
        bridge_lib/observation_encoder.h
        bridge_lib/bridge_deck.cc
)
target_link_libraries(bridge_lib PUBLIC double_dummy_solver)

pybind11_add_module(bridge SHARED bridge_lib/pybind.cc)
target_include_directories(bridge PUBLIC "third_party/pybind11/include")
target_link_libraries(bridge PUBLIC bridge_lib)
set_target_properties(bridge PROPERTIES OUTPUT_NAME "bridge")

add_library(rela_lib rela/batcher.h
        rela/batcher.cc
        rela/tensor_dict.h
        rela/utils.h
        rela/batch_runner.h
        rela/batch_runner.cc
        rela/thread_loop.h
        rela/context.cc
        rela/context.h
        rela/transition.h
        rela/transition.cc
        rela/prioritized_replay.h
        rela/logging.h
)

target_include_directories(rela_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela_lib PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(rela_lib PUBLIC ${TORCH_PYTHON_LIBRARIES})

#find_package(Python COMPONENTS Interpreter Development)
#pybind11_add_module(rela rela/pybind.cc)
pybind11_add_module(rela SHARED rela/pybind.cc)
target_include_directories(rela PUBLIC "third_party/pybind11/include")
target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})



add_library(_bridgelearn
        rlcc/bridge_env.h
        rlcc/bridge_env.cc
        rlcc/bridge_dataset.h
        rlcc/bridge_dataset.cc
        rlcc/bridge_agent.cc
        rlcc/bridge_agent.h
        rlcc/supervise_data_generator.h
        playcc/utils.h
        playcc/pimc.h
        playcc/resampler.h
        playcc/deck_sampler.h
        playcc/play_bot.h
        playcc/cheat_bot.h
        playcc/vector_utils.h
        playcc/outcome_vector.h
)
target_link_libraries(_bridgelearn PUBLIC bridge_lib)
target_link_libraries(_bridgelearn PUBLIC rela_lib)
target_include_directories(_bridgelearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(bridgelearn rlcc/pybind.cc)
target_link_libraries(bridgelearn PUBLIC _bridgelearn)
target_link_libraries(bridgelearn PUBLIC bridge_lib)
target_link_libraries(bridgelearn PUBLIC rela_lib)
target_include_directories(bridgelearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(main main.cc)
target_link_libraries(main PUBLIC bridge_lib)
target_link_libraries(main PUBLIC rela_lib)
target_link_libraries(main PUBLIC _bridgelearn)
