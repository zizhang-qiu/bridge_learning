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
#include "detailed_encoder.h"
#include "env.h"
#include "rela/logging.h"
#include "rela/tensor_dict.h"
#include "encoder_registerer.h"
#include "bridge_dataset.h"
namespace ble = bridge_learning_env;

namespace rlcc {

struct BridgeEnvOptions {
  bool bidding_phase = true;
  bool playing_phase = false;
  std::string encoder = "canonical";
  bool verbose = false;
  int max_len = 50;
};

class BridgeEnv final : public GameEnv {
 public:
  BridgeEnv(const ble::GameParameters &params, const BridgeEnvOptions &options);

  void SetBridgeDataset(std::shared_ptr<BridgeDataset> bridge_dataset) {
    bridge_dataset_ = std::move(bridge_dataset);
  }

  int MaxNumAction() const override { return game_.NumDistinctActions() + 1; }

  std::tuple<int, int, int> FeatureSize() const {
    int size = encoder_->Shape()[0];
    // Remove other players' hands.
    int priv_size = size - ble::kNumCards * (ble::kNumPlayers - 1);
    // Remove all hands.
    int publ_size = size - ble::kNumCards * ble::kNumPlayers;
    return {size, priv_size, publ_size};
  }

  void ResetWithDeck(const std::vector<int> &cards);

  void ResetWithDeckAndDoubleDummyResults(
      const std::vector<int> &cards,
      const std::vector<int> &double_dummy_results);

  bool Reset() override;

  void ResetWithDataSet();

  void Step(const ble::BridgeMove &move);

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
                   [](int i) { return static_cast<float>(i) / static_cast<float>(ble::kMaxUtility); });
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


  // Use -1 for current player.
  rela::TensorDict Feature(int player) const override;

  const ble::GameParameters &Parameters() const;

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
  std::unique_ptr<ble::ObservationEncoder> encoder_;

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
