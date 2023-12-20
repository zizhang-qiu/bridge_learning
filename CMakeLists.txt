cmake_minimum_required(VERSION 3.10.0)
project(bridge_learning VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif ()

message(STATUS CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

set(Python3_FIND_VIRTUALENV ONLY)
if (WIN32)

    message("Compiling on Windows")
    #
    #    set(PYTHON_EXECUTABLE "${CONDA_PREFIX}/python.exe")
    #    set(PYTHON_INCLUDE_DIR "${CONDA_PREFIX}/include")
    #    set(PYTHON_LIBRARIES "${CONDA_PREFIX}/libs/python38.lib")
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
    #    set(Python3_ROOT_DIR /home/work/miniconda3/envs/bridge_research/bin)
    #    set(PYTHON_EXECUTABLE /home/work/miniconda3/envs/bridge_research/bin/python)
endif ()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
#find_package(PythonLibs REQUIRED)

message(STATUS Python3_EXECUTABLE=${Python3_EXECUTABLE})
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
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
add_subdirectory(third_party/cxxopts)
add_subdirectory(third_party/spdlog)
#set(PYBIND11_FINDPYTHON ON)

# We add the subdirectory here so open_spiel_core can #include absl.
set(ABSL_PROPAGATE_CXX_STD ON)
#set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -lstdc++fs")
add_subdirectory(third_party/abseil-cpp)
include_directories(third_party/abseil-cpp)
include_directories(third_party/spdlog/include)

add_subdirectory(bridge_lib)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    pybind11_add_module(bridge SHARED pybridge/pybind.cc)
    target_include_directories(bridge PUBLIC "third_party/pybind11/include")
    target_include_directories(bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(bridge PUBLIC bridge_lib)
    set_target_properties(bridge PROPERTIES OUTPUT_NAME "bridge")
endif ()

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
target_include_directories(rela_lib PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(rela_lib PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(rela_lib PUBLIC ${TORCH_PYTHON_LIBRARIES})

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    pybind11_add_module(rela SHARED rela/pybind.cc)
    target_include_directories(rela PUBLIC "third_party/pybind11/include")
    target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(rela PUBLIC ${TORCH_INCLUDE_DIRS})
    target_include_directories(rela PUBLIC ${Python3_INCLUDE_DIRS})
    target_link_libraries(rela PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
endif ()


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
        playcc/pareto_front.h
        playcc/pareto_front.cc
        playcc/outcome_vector.cc
        playcc/alpha_mu_search.cc
        playcc/alpha_mu_search.h
        playcc/utils.cc
        playcc/pimc.cc
        playcc/resampler.cc
        playcc/worlds.cc
        playcc/worlds.h
        playcc/bridge_state_without_hidden_info.cc
        playcc/bridge_state_without_hidden_info.h
        playcc/log_utils.cc
        playcc/log_utils.h
        playcc/alpha_mu_bot.cc
        playcc/alpha_mu_bot.h
        playcc/transposition_table.cc
        playcc/transposition_table.h
        playcc/game_tree.h
)
target_link_libraries(_bridgelearn PUBLIC bridge_lib)
target_link_libraries(_bridgelearn PUBLIC rela_lib)
target_include_directories(_bridgelearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_bridgelearn PUBLIC
        absl::algorithm
        absl::flags
        absl::flags_parse
        absl::flat_hash_map
        absl::optional
        absl::random_random
        absl::str_format
        absl::strings
        absl::time)

add_executable(pareto_front_test playcc/pareto_front_test.cc)
target_link_libraries(pareto_front_test PRIVATE _bridgelearn)
add_test(pareto_front_test pareto_front_test)

add_executable(log_utils_test playcc/log_utils_test.cc)
target_link_libraries(log_utils_test PRIVATE _bridgelearn)
add_test(log_utils_test log_utils_test)

add_executable(bridge_match playcc/match.cc)
target_link_libraries(bridge_match PRIVATE _bridgelearn)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    pybind11_add_module(bridgelearn rlcc/pybind.cc)
    target_link_libraries(bridgelearn PUBLIC _bridgelearn)
    target_link_libraries(bridgelearn PUBLIC bridge_lib)
    target_link_libraries(bridgelearn PUBLIC rela_lib)
    target_include_directories(bridgelearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

add_executable(main main.cc)
target_link_libraries(main PUBLIC bridge_lib)
target_link_libraries(main PUBLIC rela_lib)
target_link_libraries(main PUBLIC _bridgelearn)


add_executable(temp test.cc)
target_link_libraries(temp PUBLIC bridge_lib)
target_link_libraries(temp PUBLIC rela_lib)
target_link_libraries(temp PUBLIC _bridgelearn)
