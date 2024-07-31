//
// Created by qzz on 2023/10/7.
//

#ifndef BRIDGE_LEARNING_RLCC_BRIDGE_ENV_H_
#define BRIDGE_LEARNING_RLCC_BRIDGE_ENV_H_
#include <algorithm>
#include <utility>

#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/dnns_encoder.h"
#include "bridge_lib/jps_encoder.h"
#include "bridge_lib/pbe_encoder.h"
#include "env.h"
#include "rela/logging.h"
#include "rela/tensor_dict.h"


#include "bridge_dataset.h"
namespace ble = bridge_learning_env;

namespace rlcc {

struct BridgeEnvOptions {
  bool bidding_phase = true;
  bool playing_phase = false;
  // Whether the feature contains pbe style.
  bool pbe_feature = false;
  // Whether the feature contains jps style.
  bool jps_feature = false;
  // Whether the feature contains dnns style.
  bool dnns_feature = false;
  bool verbose = false;
  int max_len = 50;
};

class BridgeEnv : public GameEnv {
 public:
  BridgeEnv(const ble::GameParameters& params, const BridgeEnvOptions& options);

  void SetBridgeDataset(std::shared_ptr<BridgeDataset> bridge_dataset) {
    bridge_dataset_ = std::move(bridge_dataset);
  }

  int MaxNumAction() const override { return game_.NumDistinctActions() + 1; }

  int FeatureSize() const { return encoder_.Shape()[0]; }

  void ResetWithDeck(const std::vector<int>& cards);

  void ResetWithDeckAndDoubleDummyResults(
      const std::vector<int>& cards,
      const std::vector<int>& double_dummy_results);

  bool Reset() override;

  void ResetWithDataSet();

  void Step(const ble::BridgeMove& move);

  void Step(int uid) override;

  bool Terminated() const override;

  ble::BridgeObservation BleObservation() const;

  std::vector<int> Returns() const;

  std::string ToString() const override;

  ble::Player CurrentPlayer() const override;

  int CurrentPartnership() const override {
    if (state_ == nullptr) {
      return -1;
    }
    return ble::Partnership(state_->CurrentPlayer());
  }

  float PlayerReward(int player) const override {
    RELA_CHECK_NOTNULL(state_);
    return static_cast<float>(state_->Scores()[player]);
  }

  std::vector<float> Rewards() const override {
    std::vector<int> returns = Returns();
    std::vector<float> rewards(returns.size());

    std::transform(returns.begin(), returns.end(), rewards.begin(),
                   [](int i) { return static_cast<float>(i); });
    return rewards;
  }

  ble::BridgeState BleState() const {
    RELA_CHECK_NOTNULL(state_);
    return *state_;
  }

  ble::BridgeGame BleGame() const { return game_; }

  ble::BridgeMove GetMove(int uid) const { return game_.GetMove(uid); }

  ble::Player LastActivePlayer() const { return last_active_player_; }

  ble::BridgeMove LastMove() const { return last_move_; }

  rela::TensorDict Feature(int player = -1) const override;

  const ble::GameParameters& Parameters() const;

  EnvSpec Spec() const override {
    return {ble::kNumPlayers, ble::kNumPartnerships};
  }

 private:
  rela::TensorDict TerminalFeature() const;
  const int max_len_;
  int num_step_ = 0;
  const ble::GameParameters params_;
  const ble::BridgeGame game_;
  const BridgeEnvOptions options_;
  std::unique_ptr<ble::BridgeState> state_;
  const ble::CanonicalEncoder encoder_;
  const ble::PBEEncoder pbe_encoder_;
  const ble::JPSEncoder jps_encoder_;
  const ble::DNNsEncoder dnns_encoder_;

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
}  // namespace rlcc

#endif  //BRIDGE_LEARNING_RLCC_BRIDGE_ENV_H_
