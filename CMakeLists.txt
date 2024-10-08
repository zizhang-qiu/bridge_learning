cmake_minimum_required(VERSION 3.10.0)
project(bridge_learning VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif ()

message(STATUS CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
if (WIN32)
    add_definitions(-DNOMINMAX)  # Define NOMINMAX for Windows builds

endif ()

#set(Python3_ROOT_DIR D:/Anaconda/envs/bridge_research)
#set(Python3_FIND_VIRTUALENV ONLY)
if (WIN32)
    set(CUDA_HOME "D:/cuda-12.5")
#    set(CMAKE_GENERATOR_TOOLSET "D:/cuda-12.5/")
#    set(CUDA_TOOLKIT_ROOT_DIR "D:/cuda-12.5")
#    set(CMAKE_CUDA_COMPILER "D:/cuda-12.5/bin/nvcc.exe")
#    set(CUDA_CUDART_LIBRARY "D:\\cuda-12.5\\lib\\x64\\cudart.lib")
    message("Compiling on Windows")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYBIND11_COMPILER_TYPE=\\\"_msvc\\\" -DPYBIND11_STDLIB=\"\" -DPYBIND11_BUILD_ABI=\"\"")

    #
    #    set(PYTHON_EXECUTABLE "${CONDA_PREFIX}/python.exe")
    #    set(PYTHON_INCLUDE_DIR "${CONDA_PREFIX}/include")
    #    set(PYTHON_LIBRARIES "${CONDA_PREFIX}/libs/python38.lib")
else ()
    message("Compiling on Linux")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYBIND11_COMPILER_TYPE=\\\"_gcc\\\" -DPYBIND11_STDLIB=\\\"_libstdcpp\\\" -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
    set(CMAKE_CUDA_FLAGS "-std=c++17")
    SET(CUDA_HOME /usr/local/cuda)
    #    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
    #    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.6")
    #    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda-11.8)
    #    set(CUDA_NVCC_EXECUTABLE /usr/local/cuda-11.6/bin/nvcc)
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

add_library(rela_lib
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
        rela/types.h
        rela/prioritized_replay2.cc
        rela/prioritized_replay2.h
        rela/batcher.cc
        rela/batcher.h
        rela/r2d2.h
)

target_include_directories(rela_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela_lib PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(rela_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rela_lib PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(rela_lib PUBLIC ${TORCH_PYTHON_LIBRARIES})
target_compile_definitions(rela_lib PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    pybind11_add_module(pyrela SHARED rela/pybind.cc)
    target_include_directories(pyrela PUBLIC "third_party/pybind11/include")
    target_include_directories(pyrela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(pyrela PUBLIC ${TORCH_INCLUDE_DIRS})
    target_include_directories(pyrela PUBLIC ${Python3_INCLUDE_DIRS})
    target_link_libraries(pyrela PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
    target_compile_definitions(pyrela PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif ()


add_library(_bridgelearn OBJECT
        rlcc/env.h
        rlcc/bridge_env.h
        rlcc/bridge_env.cc
        rlcc/bridge_dataset.h
        rlcc/bridge_dataset.cc
        rlcc/supervise_data_generator.h
        rlcc/belief_data_gen.cc
        rlcc/belief_data_gen.h
        rlcc/bridge_actor.h
        rlcc/bridge_actor.cc
        rlcc/actor_based_belief_gen.cc
        rlcc/actor_based_belief_gen.h
        rlcc/duplicate_env.cc
        rlcc/duplicate_env.h
        rlcc/utils.h
        rlcc/rnn_buffer.h
        rlcc/clone_data_generator.cc
        rlcc/clone_data_generator.h
        rlcc/detailed_encoder.cc
        rlcc/detailed_encoder.h
        rlcc/encoder_registerer.h
        rlcc/encoder_registerer.cc
        rlcc/utils.cc
        rlcc/bridge_env_new.h
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
target_compile_definitions(_bridgelearn PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

add_library(_bridgeplay
        playcc/utils.h
        playcc/pimc.h
        playcc/resampler.h
        playcc/deck_sampler.h
        playcc/play_bot.h
        playcc/play_bot.cc
        playcc/dds_bot.h
        playcc/vector_utils.h
        playcc/outcome_vector.h
        playcc/pareto_front.h
        playcc/pareto_front.cc
        playcc/outcome_vector.cc
        playcc/utils.cc
        playcc/pimc.cc
        playcc/resampler.cc
        playcc/worlds.cc
        playcc/worlds.h
        playcc/bridge_state_without_hidden_info.cc
        playcc/bridge_state_without_hidden_info.h
        playcc/common_utils/log_utils.cc
        playcc/common_utils/log_utils.h
        playcc/common_utils/log_utils.cc
        playcc/common_utils/log_utils.h
        playcc/alpha_mu_bot.cc
        playcc/alpha_mu_bot.h
        playcc/transposition_table.cc
        playcc/transposition_table.h
        playcc/common_utils/logger.h
        playcc/common_utils/file.cc
        playcc/common_utils/file.h
        playcc/common_utils/logger.h
        playcc/common_utils/file.cc
        playcc/common_utils/file.h
        playcc/deal_analyzer.cc
        playcc/deal_analyzer.h
        playcc/common_utils/stat_manager.h
        playcc/common_utils/threaded_queue.h
        playcc/common_utils/stat_manager.h
        playcc/common_utils/threaded_queue.h
        playcc/dds_bot.cc
        playcc/torch_actor.cc
        playcc/torch_actor.h
        playcc/nn_belief_resampler.cc
        playcc/nn_belief_resampler.h
        playcc/belief_based_opening_lead_bot.cc
        playcc/belief_based_opening_lead_bot.h
        playcc/nn_belief_resampler.cc
        playcc/nn_belief_resampler.h
        playcc/belief_based_opening_lead_bot.cc
        playcc/belief_based_opening_lead_bot.h
        playcc/opening_lead_evaluation_thread_loop.cc
        playcc/opening_lead_evaluation_thread_loop.h
        playcc/dds_evaluator.cc
        playcc/dds_evaluator.h
        playcc/wbridge5_trajectory_bot.h
        playcc/rule_based_defender.cc
        playcc/rule_based_defender.h
        playcc/trajectory_bidding_bot.cc
)
target_compile_definitions(_bridgeplay PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
# target_link_directories(_bridgeplay PUBLIC bridge_lib/third_party/dds/src)
# target_link_libraries(_bridgeplay PUBLIC dds)
target_link_libraries(_bridgeplay PUBLIC bridge_lib)
target_link_libraries(_bridgeplay PUBLIC rela_lib)

target_link_libraries(_bridgeplay PUBLIC
        absl::algorithm
        absl::flags
        absl::flags_parse
        absl::flat_hash_map
        absl::optional
        absl::random_random
        absl::str_format
        absl::strings
        absl::time
        ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
target_include_directories(_bridgeplay PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(_bridgeplay PUBLIC bridge_lib/third_party/dds/include)
#add_executable(pareto_front_test playcc/unit_tests/pareto_front_test.cc)
#target_link_libraries(pareto_front_test PRIVATE _bridgeplay)
#add_test(pareto_front_test pareto_front_test)
#
#add_executable(log_utils_test playcc/unit_tests/log_utils_test.cc)
#target_link_libraries(log_utils_test PRIVATE _bridgeplay)
#add_test(log_utils_test log_utils_test)
#
#add_executable(transposition_table_test playcc/unit_tests/transposition_table_test.cc)
#target_link_libraries(transposition_table_test PRIVATE _bridgeplay)
#add_test(transposition_table_test transposition_table_test)
#
#add_executable(bridge_match playcc/match.cc)
#target_link_libraries(bridge_match PRIVATE _bridgeplay)






if (CMAKE_BUILD_TYPE STREQUAL "Release")
    pybind11_add_module(bridgelearn SHARED rlcc/pybind.cc)
    target_link_libraries(bridgelearn PUBLIC _bridgelearn)
    target_link_libraries(bridgelearn PUBLIC bridge_lib)
    target_link_libraries(bridgelearn PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
    target_include_directories(bridgelearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
    target_compile_definitions(bridgelearn PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

    pybind11_add_module(bridgeplay SHARED playcc/pybind.cc)
    target_link_libraries(bridgeplay PUBLIC bridge_lib)
    target_link_libraries(bridgeplay PUBLIC _bridgeplay)
    target_link_libraries(bridgeplay PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
    target_include_directories(bridgeplay PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(bridgeplay PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

    add_executable(temp test.cc)


    target_link_libraries(temp PUBLIC bridge_lib)
    # target_link_libraries(temp PUBLIC bridge_lib/third_party/dds/src/dds)
    target_link_libraries(temp PUBLIC rela_lib)
    target_link_libraries(temp PUBLIC _bridgeplay)
    target_link_libraries(temp PUBLIC _bridgelearn)
    # target_link_directories(temp PUBLIC bridge_lib/third_party/dds/src)
    # target_link_libraries(temp PRIVATE dds)
    target_link_libraries(temp PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES} ${Python3_LIBRARIES})
    target_include_directories(temp PUBLIC ${Python3_INCLUDE_DIRS})
    target_include_directories(temp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

#add_executable(main main.cc)
#target_link_libraries(main PUBLIC bridge_lib)
#target_link_libraries(main PUBLIC rela_lib)
#target_link_libraries(main PUBLIC _bridgeplay)
#
#
#
#
#
#add_executable(temp2 temp2.cc)
#target_link_libraries(temp2 PUBLIC bridge_lib)
#target_link_libraries(temp2 PUBLIC rela_lib)
#target_link_libraries(temp2 PUBLIC _bridgeplay)
#
#add_executable(analyze_from_traj_and_tt playcc/analyze_from_traj_and_tt.cc)
#target_link_libraries(analyze_from_traj_and_tt PUBLIC bridge_lib)
#target_link_libraries(analyze_from_traj_and_tt PUBLIC rela_lib)
#target_link_libraries(analyze_from_traj_and_tt PUBLIC _bridgeplay)


