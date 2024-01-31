//
// Created by qzz on 2023/10/7.
//

#ifndef BRIDGE_LEARNING_RLCC_BRIDGE_ENV_H_
#define BRIDGE_LEARNING_RLCC_BRIDGE_ENV_H_
#include <utility>

#include "rela/logging.h"
#include "rela/tensor_dict.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_game.h"
#include "bridge_lib/canonical_encoder.h"
#include "bridge_dataset.h"
namespace ble = bridge_learning_env;
namespace rlcc {

struct BridgeEnvOptions {
  bool bidding_phase = true;
  bool playing_phase = false;

  bool verbose = false;
};

class BridgeEnv {
 public:
  BridgeEnv(const ble::GameParameters &params, const BridgeEnvOptions& options);

  void SetBridgeDataset(std::shared_ptr<BridgeDataset> bridge_dataset) {
    bridge_dataset_ = std::move(bridge_dataset);
  }

  int FeatureSize() const { return encoder_.Shape()[0]; }

  void ResetWithDeck(const std::vector<int> &cards);

  void ResetWithDeckAndDoubleDummyResults(const std::vector<int> &cards, const std::vector<int> &double_dummy_results);

  void Reset();

  void ResetWithDataSet();

  void Step(const ble::BridgeMove &move);

  void Step(int uid);

  bool Terminated() const;

  ble::BridgeObservation BleObservation() const;

  std::vector<int> Returns() const;

  std::string ToString() const;

  ble::Player CurrentPlayer() const;

  ble::BridgeState BleState() const {
    RELA_CHECK_NOTNULL(state_);
    return *state_;
  }

  ble::BridgeGame BleGame() const {
    return game_;
  }

  ble::BridgeMove GetMove(int uid) const {
    return game_.GetMove(uid);
  }

  ble::Player LastActivePlayer() const {
    return last_active_player_;
  }

  ble::BridgeMove LastMove() const {
    return last_move_;
  }

  rela::TensorDict Feature() const;

  const ble::GameParameters &Parameters() const;

 private:
  rela::TensorDict TerminalFeature() const;
  const ble::GameParameters params_;
  const ble::BridgeGame game_;
  const BridgeEnvOptions options_;
  std::unique_ptr<ble::BridgeState> state_;
  const ble::CanonicalEncoder encoder_;

  ble::Player last_active_player_;
  ble::BridgeMove last_move_;

  std::shared_ptr<BridgeDataset> bridge_dataset_ = nullptr;

};

class BridgeVecEnv {
 public:
  BridgeVecEnv() = default;

  int Size() const { return static_cast<int>(envs_.size()); }

  void Append(std::shared_ptr<BridgeEnv> env) {
    envs_.push_back(std::move(env));
  }

  void Reset();

  bool AnyTerminated() const;

  bool AllTerminated() const;

  void Step(rela::TensorDict reply);

  rela::TensorDict Feature() const;

  void DisPlay(int num_envs) const;

 private:
  std::vector<std::shared_ptr<BridgeEnv>> envs_;
};
}

#endif //BRIDGE_LEARNING_RLCC_BRIDGE_ENV_H_
