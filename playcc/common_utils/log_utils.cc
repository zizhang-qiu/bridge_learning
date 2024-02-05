//
// Created by qzz on 2023/12/3.
//

#include "log_utils.h"
#include <iostream>

void SpielDefaultErrorHandler(const std::string& error_msg) {
  std::cerr << "Spiel Fatal Error: " << error_msg << std::endl
            << std::endl
            << std::flush;
  std::exit(1);
}

ErrorHandler error_handler = SpielDefaultErrorHandler;

void SetErrorHandler(ErrorHandler new_error_handler) {
  error_handler = new_error_handler;
}

void SpielFatalError(const std::string& error_msg) {
  error_handler(error_msg);
  // The error handler should not return. If it does, we will abort the process.
  std::cerr << "Error handler failure - exiting" << std::endl;
  std::exit(1);
}
