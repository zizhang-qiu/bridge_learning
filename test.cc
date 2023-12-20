//
// Created by qzz on 2023/12/16.
//
#include <iostream>

#include "third_party/cxxopts/include/cxxopts.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "absl/strings/str_format.h"

void rotating_example() {
  // Create a file rotating logger with 5 MB size max and 3 rotated files
  auto max_size = 1048576 * 5;
  auto max_files = 3;
  auto logger = spdlog::rotating_logger_mt("some_logger_name", "logs/rotating.txt", max_size, max_files);
}
void basic_logfile_example() {
  try {
    auto logger = spdlog::basic_logger_mt("basic_logger", "logs/basic-log.txt");
    logger->error("This is an error.");
  }
  catch (const spdlog::spdlog_ex &ex) {
    std::cout << "Log init failed: " << ex.what() << std::endl;
  }
}
void stdout_example() {
  // create a color multi-threaded logger
  auto console = spdlog::stdout_color_mt("console");
  auto err_logger = spdlog::stderr_color_mt("stderr");
  spdlog::get("console")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");
}
int main(int argc, char** argv) {
//  spdlog::info("Welcome to spdlog!");
//  spdlog::error("Some error message with arg: {}", 1);
//
//  spdlog::warn("Easy padding in numbers like {:08d}", 12);
//  spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
//  spdlog::info("Support for floats {:03.2f}", 1.23456);
//  spdlog::info("Positional args are {1} {0}..", "too", "supported");
//  spdlog::info("{:<30}", "left aligned");
//
//  spdlog::set_level(spdlog::level::debug); // Set global log level to debug
//  spdlog::debug("This message should be displayed..");
//
//  // change log pattern
//  spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
//
//  // Compile time log levels
//  // define SPDLOG_ACTIVE_LEVEL to desired level
//  SPDLOG_TRACE("Some trace message with param {}", 42);
//  SPDLOG_DEBUG("Some debug message");

//  stdout_example();
//  basic_logfile_example();
  cxxopts::Options options("Bridge Play Match", "A program plays bridge between alphamu and pimc.");
  options.add_options()
      ("m, num_max_moves", "Number of max moves in alphamu search", cxxopts::value<int>()->default_value("1"))
      ("w, num_worlds", "Number of possible worlds", cxxopts::value<int>()->default_value("20"))
      ("num_deals", "Number of deals with different results", cxxopts::value<int>()->default_value("200"))
      ("contract", "The contract of the deals", cxxopts::value<std::string>()->default_value("3NT"))
      ("seed", "Random seed for generating deals", cxxopts::value<int>()->default_value("66"))
      ("show_play", "Whether to show the played games", cxxopts::value<bool>()->default_value("false"));

  auto s = absl::StrFormat("I want %d", 1);
  auto result = options.parse(argc, argv);
  spdlog::info("Match start.");
  auto logger = spdlog::basic_logger_mt("basic_logger", "D:/Projects/bridge/log2.txt");
  logger->info("Match start.");
}