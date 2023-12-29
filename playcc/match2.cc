//
// Created by qzz on 2023/12/29.
//
#include "alpha_mu_bot.h"
#include "cheat_bot.h"
#include "duplicate_threadloop.h"
#include "pimc.h"
#include "rela/context.h"

int main() {
  EvaluationConfig cfg{5};
  const AlphaMuConfig alpha_mu_cfg{2, 20, false, true, false, true};
  const PIMCConfig pimc_cfg{20, false};
  std::shared_ptr<Resampler> resampler = std::make_shared<UniformResampler>(1);
  std::shared_ptr<PlayBot> player1 = std::make_shared<AlphaMuBot>(resampler, alpha_mu_cfg);
  std::shared_ptr<PlayBot> player2 = std::make_shared<PIMCBot>(resampler, pimc_cfg);
  std::shared_ptr<PlayBot> defender = std::make_shared<CheatBot>();
  std::shared_ptr<ThreadedQueue<EvaluationResult>> queue = std::make_shared<ThreadedQueue<EvaluationResult>>(1000000);

  auto context = std::make_shared<rela::Context>();
  int num_threads = 2;
  for (int i = 0; i < num_threads; ++i) {
    auto t = std::make_shared<DuplicateThreadloop>(cfg, resampler, player1, player2, defender, queue);
    context->pushThreadLoop(t);
  }
  context->start();
  while(queue->Size() < cfg.num_deals * num_threads) {
    Sleep(1000);
  }
  std::vector<EvaluationResult> results;
  while (!queue->Empty()) {
    results.push_back(queue->Pop().value());
  }

  rela::utils::printVector(results);
}
