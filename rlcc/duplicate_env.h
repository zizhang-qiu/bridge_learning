#ifndef RLCC_DUPLICATE_ENV_H
#define RLCC_DUPLICATE_ENV_H

#include <array>
#include <memory>

#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/dnns_encoder.h"
#include "bridge_lib/jps_encoder.h"
#include "bridge_lib/pbe_encoder.h"
#include "env.h"
#include "rela/utils.h"
#include "rlcc/bridge_dataset.h"
#include "rlcc/bridge_env.h"

namespace ble = bridge_learning_env;

namespace rlcc {
class DuplicateEnv : public GameEnv {
 public:
  DuplicateEnv(const ble::GameParameters &params,
               const BridgeEnvOptions &options)
      : game_(params),
        options_(options) {
    if (!options.bidding_phase && !options.playing_phase) {
      rela::utils::RelaFatalError(
          "Both bidding and playing phase are off. At least one phase should "
          "be on.");
    }
    encoder_ = rlcc::LoadEncoder(options_.encoder, std::make_shared<ble::BridgeGame>(game_), {});
  }

  DuplicateEnv(const ble::GameParameters &params,
               const BridgeEnvOptions &options,
               const std::shared_ptr<BridgeDataset> &dataset)
      : game_(params),
        options_(options),
        dataset_(dataset) {
    if (!options.bidding_phase && !options.playing_phase) {
      rela::utils::RelaFatalError(
          "Both bidding and playing phase are off. At least one phase should "
          "be on.");
    }
    encoder_ = rlcc::LoadEncoder(options_.encoder, std::make_shared<ble::BridgeGame>(game_));
  }

  DuplicateEnv(const DuplicateEnv &env)
      : game_index_(env.game_index_),
        game_(env.game_),
        options_(env.options_),
        terminated_(env.terminated_),
        dataset_(env.dataset_) {
    states_[0] = std::make_unique<ble::BridgeState>(*env.states_[0]);
    states_[1] = std::make_unique<ble::BridgeState>(*env.states_[1]);
    encoder_ = rlcc::LoadEncoder(options_.encoder, std::make_shared<ble::BridgeGame>(game_));
  }

  void SetBridgeDataset(const std::shared_ptr<BridgeDataset> &bridge_dataset) {
    dataset_ = bridge_dataset;
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

  bool Reset() override {
    num_steps_ = 0;
    if (dataset_ != nullptr) {
      return ResetWithDataset();
    }
    return ResetWithoutDataset();
  }

  void Step(int uid) override;

  bool Terminated() const override {
    return terminated_;
  }

  int CurrentPlayer() const override {
    if (game_index_ == 0) {
      return states_[0]->CurrentPlayer();
    }

    return (states_[1]->CurrentPlayer() - 1 + ble::kNumPlayers) % ble::kNumPlayers;
  }

  float PlayerReward(int player) const override;

  std::vector<float> Rewards() const override;

  std::string ToString() const override;

  int GameIndex() const { return game_index_; }

  int CurrentPartnership() const override {
    if (game_index_ == 0) {
      return ble::Partnership(states_[0]->CurrentPlayer());
    }
    return 1 - ble::Partnership(states_[1]->CurrentPlayer());
  }

  std::vector<int> LegalActions() const override;

  EnvSpec Spec() const override {
    return {ble::kNumPlayers, ble::kNumPartnerships};
  }

  rela::TensorDict Feature(int player) const override;

 private:
  bool ResetWithoutDataset();
  bool ResetWithDataset();
  int game_index_ = 0;
  const ble::BridgeGame game_;
  const BridgeEnvOptions options_;
  int num_steps_ = 0;
  // 2 tables.
  std::array<std::unique_ptr<ble::BridgeState>, 2> states_;
  bool terminated_ = true;
  std::shared_ptr<BridgeDataset> dataset_ = nullptr;

  std::unique_ptr<ble::ObservationEncoder> encoder_;
};
}  // namespace rlcc

#endif /* RLCC_DUPLICATE_ENV_H */
