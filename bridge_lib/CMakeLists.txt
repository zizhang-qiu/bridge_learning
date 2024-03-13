cmake_minimum_required(VERSION 3.10.0)
project(bridge_learning_env VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(double_dummy_solver OBJECT
        third_party/dds/include/dll.h
        third_party/dds/include/portab.h
        third_party/dds/src/ABsearch.cpp
        third_party/dds/src/ABsearch.h
        third_party/dds/src/ABstats.cpp
        third_party/dds/src/ABstats.h
        third_party/dds/src/CalcTables.cpp
        third_party/dds/src/CalcTables.h
        third_party/dds/src/dds.cpp
        third_party/dds/src/dds.h
        third_party/dds/src/DealerPar.cpp
        third_party/dds/src/debug.h
        third_party/dds/src/dump.cpp
        third_party/dds/src/dump.h
        third_party/dds/src/File.cpp
        third_party/dds/src/File.h
        third_party/dds/src/Init.cpp
        third_party/dds/src/Init.h
        third_party/dds/src/LaterTricks.cpp
        third_party/dds/src/LaterTricks.h
        third_party/dds/src/Memory.cpp
        third_party/dds/src/Memory.h
        third_party/dds/src/Moves.cpp
        third_party/dds/src/Moves.h
        third_party/dds/src/Par.cpp
        third_party/dds/src/parallel.h
        third_party/dds/src/PBN.cpp
        third_party/dds/src/PBN.h
        third_party/dds/src/PlayAnalyser.cpp
        third_party/dds/src/PlayAnalyser.h
        third_party/dds/src/QuickTricks.cpp
        third_party/dds/src/QuickTricks.h
        third_party/dds/src/Scheduler.cpp
        third_party/dds/src/Scheduler.h
        third_party/dds/src/SolveBoard.cpp
        third_party/dds/src/SolveBoard.h
        third_party/dds/src/SolverIF.cpp
        third_party/dds/src/SolverIF.h
        third_party/dds/src/System.cpp
        third_party/dds/src/System.h
        third_party/dds/src/ThreadMgr.cpp
        third_party/dds/src/ThreadMgr.h
        third_party/dds/src/Timer.cpp
        third_party/dds/src/Timer.h
        third_party/dds/src/TimerGroup.cpp
        third_party/dds/src/TimerGroup.h
        third_party/dds/src/TimerList.cpp
        third_party/dds/src/TimerList.h
        third_party/dds/src/TimeStat.cpp
        third_party/dds/src/TimeStat.h
        third_party/dds/src/TimeStatList.cpp
        third_party/dds/src/TimeStatList.h
        third_party/dds/src/TransTable.h
        third_party/dds/src/TransTableL.cpp
        third_party/dds/src/TransTableL.h
        third_party/dds/src/TransTableS.cpp
        third_party/dds/src/TransTableS.h
)
target_compile_definitions(double_dummy_solver PUBLIC DDS_THREADS_STL)
include_directories(third_party/dds/include)

add_library(bridge_lib
        bridge_card.cc
        bridge_card.h
        bridge_hand.cc
        bridge_hand.h
        bridge_move.cc
        bridge_move.h
        bridge_utils.cc
        bridge_utils.h
        bridge_game.cc
        bridge_game.h
        bridge_deck.cc
        bridge_deck.h
        bridge_scoring.cc
        bridge_scoring.h
        auction_tracker.cc
        auction_tracker.h
        bridge_history_item.cc
        bridge_history_item.h
        trick.cc
        trick.h
        bridge_state.cc
        bridge_state.h
        bridge_observation.cc
        bridge_observation.h
        observation_encoder.h
        canonical_encoder.cc
        canonical_encoder.h
        pbe_encoder.cc
        pbe_encoder.h
        jps_encoder.cc
        jps_encoder.h
        dnns_encoder.cc
        dnns_encoder.h
        utils.cc
        utils.h
)
target_include_directories(bridge_lib PUBLIC third_party/dds/include)
# target_link_directories(bridge_lib PUBLIC third_party/dds/src)
target_link_libraries(bridge_lib PUBLIC double_dummy_solver)

add_executable(bridge_test bridge_test.cc)
target_link_libraries(bridge_test PRIVATE bridge_lib)
add_test(bridge_test bridge_test)