//
// Created by qzz on 2023/9/19.
//

#include "context.h"

namespace rela {

Context::~Context() {
  if (!terminated()) {
    for (auto& v : loops_) {
      v->terminate();
    }
  }

  for (auto& v : threads_) {
    if (v.joinable()) {
      v.join();
    }
  }
}

int Context::pushThreadLoop(std::shared_ptr<ThreadLoop> env) {
  assert(!started_);
  loops_.push_back(std::move(env));
  return (int)loops_.size();
}

void Context::start() {
  for (int i = 0; i < (int)loops_.size(); ++i) {
    threads_.emplace_back([this, i]() {
      loops_[i]->mainLoop();
      ++numTerminatedThread_;
    });
  }
}

void Context::pause() {
  for (auto& v : loops_) {
    v->pause();
  }
}

void Context::resume() {
  for (auto& v : loops_) {
    v->resume();
  }
}

void Context::join() {
  for (auto& v : threads_) {
    v.join();
  }
  assert(terminated());
}

bool Context::terminated() {
  return numTerminatedThread_ == (int)loops_.size();
}

} // rela