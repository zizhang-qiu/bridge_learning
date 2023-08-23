cmake_minimum_required(VERSION 3.0.0)
project(bridge_lib VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()

add_library(bridge_lib bridge_lib/bridge_utils.cc bridge_lib/bridge_game.cc bridge_lib/utils.cc bridge_lib/bridge_scoring.cc bridge_lib/bridge_state.cc bridge_lib/trick.cc)
add_executable(main main.cc)
target_link_libraries(main PUBLIC bridge_lib)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
