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
  DuplicateEnv(const ble::GameParameters& params,
               const BridgeEnvOptions& options)
      : game_(params),
        options_(options),
        encoder_(std::make_shared<ble::BridgeGame>(game_)),
        pbe_encoder_(std::make_shared<ble::BridgeGame>(game_)),
        jps_encoder_(std::make_shared<ble::BridgeGame>(game_)),
        dnns_encoder_() {
    if (!options.bidding_phase && !options.playing_phase) {
      rela::utils::RelaFatalError(
          "Both bidding and playing phase are off. At least one phase should "
          "be on.");
    }
  }

  DuplicateEnv(const ble::GameParameters& params,
               const BridgeEnvOptions& options,
               const std::shared_ptr<BridgeDataset>& dataset)
      : game_(params),
        options_(options),
        dataset_(dataset),
        encoder_(std::make_shared<ble::BridgeGame>(game_)),
        pbe_encoder_(std::make_shared<ble::BridgeGame>(game_)),
        jps_encoder_(std::make_shared<ble::BridgeGame>(game_)),
        dnns_encoder_() {
    if (!options.bidding_phase && !options.playing_phase) {
      rela::utils::RelaFatalError(
          "Both bidding and playing phase are off. At least one phase should "
          "be on.");
    }
  }

  DuplicateEnv(const DuplicateEnv& env)
      : game_index_(env.game_index_),
        game_(env.game_),
        options_(env.options_),
        terminated_(env.terminated_),
        dataset_(env.dataset_),
        encoder_(env.encoder_),
        pbe_encoder_(env.pbe_encoder_),
        jps_encoder_(env.jps_encoder_),
        dnns_encoder_(env.dnns_encoder_) {
    states_[0] = std::make_unique<ble::BridgeState>(*env.states_[0]);
    states_[1] = std::make_unique<ble::BridgeState>(*env.states_[1]);
  }

  void SetBridgeDataset(const std::shared_ptr<BridgeDataset>& bridge_dataset) {
    dataset_ = bridge_dataset;
  }

  int MaxNumAction() const override { return game_.NumDistinctActions(); }

  bool Reset() override {
    if (dataset_ != nullptr) {
      return ResetWithDataset();
    }
    return ResetWithoutDataset();
  }

  void Step(int uid) override;

  bool Terminated() const override { return terminated_; }

  int CurrentPlayer() const override {
    if (game_index_ == 0) {
      return states_[0]->CurrentPlayer();
    }

    return (states_[1]->CurrentPlayer() + 1) % ble::kNumPlayers;
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

  rela::TensorDict Feature(int player=-1) const override;

 private:
  bool ResetWithoutDataset();
  bool ResetWithDataset();
  int game_index_ = 0;
  const ble::BridgeGame game_;
  const BridgeEnvOptions& options_;
  std::array<std::unique_ptr<ble::BridgeState>, 2> states_;
  bool terminated_ = true;
  std::shared_ptr<BridgeDataset> dataset_ = nullptr;

  const ble::CanonicalEncoder encoder_;
  const ble::PBEEncoder pbe_encoder_;
  const ble::JPSEncoder jps_encoder_;
  const ble::DNNsEncoder dnns_encoder_;
};
}  // namespace rlcc

#endif /* RLCC_DUPLICATE_ENV_H */
