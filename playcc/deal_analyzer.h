//
// Created by qzz on 2023/12/22.
//

#ifndef BRIDGE_LEARNING_PLAYCC_DEAL_ANALYZER_H_
#define BRIDGE_LEARNING_PLAYCC_DEAL_ANALYZER_H_
#include "pimc.h"
#include "alpha_mu_bot.h"
#include "file.h"

namespace ble = bridge_learning_env;

std::vector<ble::BridgeMove> DDSMoves(const ble::BridgeState &state);

// A deal analyzer analyzes a deal played by alphamu bot and dds
// to find which move is not optimal.
class DealAnalyzer {

 public:
  DealAnalyzer(const std::string &save_dir) : save_dir_(save_dir) {
    if (!file::Exists(save_dir)){
      file::Mkdir(save_dir);
    }
  }

  void Analyze(ble::BridgeState state, AlphaMuBot alpha_mu_bot, PIMCBot pimc_bot);

 private:
  const std::string save_dir_;

};
#endif //BRIDGE_LEARNING_PLAYCC_DEAL_ANALYZER_H_
