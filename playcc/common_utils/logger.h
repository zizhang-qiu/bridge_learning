//
// Created by qzz on 2023/12/20.
//

#ifndef BRIDGE_LEARNING_PLAYCC_LOGGER_H_
#define BRIDGE_LEARNING_PLAYCC_LOGGER_H_

#include <cstdio>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "file.h"

class Logger {
 public:
  virtual ~Logger() = default;
  virtual void Print(const std::string& str) = 0;

  // A specialization of Print that passes everything through StrFormat first.
  template <typename Arg1, typename... Args>
  void Print(const absl::FormatSpec<Arg1, Args...>& format, const Arg1& arg1,
             const Args&... args) {
    Print(absl::StrFormat(format, arg1, args...));
  }
};


// A logger to print stuff to a file.
class FileLogger : public Logger {
 public:
  FileLogger(const std::string& path, const std::string& name,
             const std::string& mode = "w")
      : fd_(absl::StrFormat("%s/log-%s.txt", path, name), mode),
        tz_(absl::LocalTimeZone()) {
    Print("%s started", name);
  }

  using Logger::Print;
  void Print(const std::string& str) override {
    const std::string time =
        absl::FormatTime("%Y-%m-%d %H:%M:%E3S", absl::Now(), tz_);
    fd_.Write(absl::StrFormat("[%s] %s\n", time, str));
    fd_.Flush();
  }

  ~FileLogger() override { Print("Closing the log."); }

 private:
  file::File fd_;
  absl::TimeZone tz_;
};


class NoopLogger : public Logger {
 public:
  using Logger::Print;
  void Print(const std::string& str) override {}
};

#endif //BRIDGE_LEARNING_PLAYCC_LOGGER_H_
