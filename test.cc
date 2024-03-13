//
// Created by qzz on 2023/12/16.
//
#include <ios>
#include <iostream>
#include <memory>
#include <random>

// #include "torch/torch.h"

// #include "rela/batcher.h"

#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_move.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/example_cards_ddts.h"
#include "bridge_lib/utils.h"
// #include "dll.h"
#include "playcc/alpha_mu_bot.h"
#include "playcc/dds_evaluator.h"
#include "playcc/pimc.h"
#include "playcc/resampler.h"
#include "playcc/utils.h"
#include "rlcc/bridge_actor.h"
#include "rlcc/bridge_dataset.h"
#include "rlcc/bridge_env.h"
#include "rlcc/bridge_env_actor.h"
#include "rlcc/duplicate_env.h"
#include "rlcc/env_actor.h"

// #include "rlcc/belief_data_gen.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

std::vector<size_t> FindNonZeroIndices(const std::vector<int>& vec) {
  std::vector<size_t> indices;

  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      indices.push_back(i);
    }
  }

  return indices;
}

int main(int argc, char** argv) {
  rlcc::BridgeEnvOptions options{};
  options.dnns_feature = true;
  options.jps_feature = true;
  options.dnns_feature = true;
  // auto dataset = std::make_shared<rlcc::BridgeDataset>(ble::example_deals,
  //                                                      ble::example_ddts);
  // rlcc::DuplicateEnv env{{}, options, dataset};
  // env.Reset();
  // std::cout << env.ToString() << std::endl;
  // for (const int action : {55, 52, 52, 52}) {
  //   std::cout << env.CurrentPlayer() << std::endl;
  //   std::cout << env.LegalActions() << std::endl;
  //   env.Step(action);
  // }

  // auto actor = std::make_shared<rlcc::AllPassActor>();
  // const rlcc::EnvActorOptions env_actor_options{};
  // rlcc::BridgeEnvActor env_actor(std::make_shared<rlcc::DuplicateEnv>(env),
  //                                env_actor_options,
  //                                {actor, actor, actor, actor});
  
  // env_actor.ObserveAfterAct();
  // env_actor.Act();
  // env_actor.ObserveBeforeAct();
  // env_actor.SendExperience();
  // env_actor.PostSendExperience();

  // std::cout << env_actor.GetEnv()->ToString() << std::endl;
}