#include "duplicate_env.h"
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>
#include <memory>
#include <vector>
#include "absl/strings/str_cat.h"
#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_scoring.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/utils.h"
#include "rela/logging.h"
#include "rela/types.h"

namespace rlcc {

bool DuplicateEnv::ResetWithoutDataset() {
  RELA_CHECK(options_.bidding_phase,
             "If you want to train only playing, a dataset contains bidding "
             "should be provided.");
  const auto deal = ble::Permutation(ble::kNumCards);
  states_[0] = std::make_unique<ble::BridgeState>(
      std::make_shared<ble::BridgeGame>(game_));
  states_[1] = std::make_unique<ble::BridgeState>(
      std::make_shared<ble::BridgeGame>(game_));

  for (const int uid : deal) {
    const auto chance = game_.GetChanceOutcome(uid);
    states_[0]->ApplyMove(chance);
    states_[1]->ApplyMove(chance);
  }
  terminated_ = false;
  game_index_ = 0;
  return true;
}

bool DuplicateEnv::ResetWithDataset() {
  const auto data = dataset_->Next();

  states_[0] = std::make_unique<ble::BridgeState>(
      std::make_shared<ble::BridgeGame>(game_));
  states_[1] = std::make_unique<ble::BridgeState>(
      std::make_shared<ble::BridgeGame>(game_));

  for (int i = 0; i < ble::kNumCards; ++i) {
    const int uid = data.deal[i];
    const auto chance = game_.GetChanceOutcome(uid);
    states_[0]->ApplyMove(chance);
    states_[1]->ApplyMove(chance);
  }
  if (data.ddt.has_value()) {
    states_[0]->SetDoubleDummyResults(data.ddt.value());
    states_[1]->SetDoubleDummyResults(data.ddt.value());
  }

  if (!options_.bidding_phase) {
    RELA_CHECK_GE(data.deal.size(), game_.MinGameLength());
    // Train playing.
    for (int i = ble::kNumCards; i < data.deal.size(); ++i) {
      const int uid = data.deal.at(i);
      const auto move = game_.GetMove(uid);
      states_[0]->ApplyMove(move);
      states_[1]->ApplyMove(move);
    }
    // It is possible that some deals in the dataset is passed out.
    // If so, we go to next deal.
    if (states_[0]->CurrentPhase() == ble::Phase::kGameOver) {
      return Reset();
    }
    RELA_CHECK(states_[0]->IsInPhase(ble::Phase::kPlay));
    RELA_CHECK(states_[1]->IsInPhase(ble::Phase::kPlay));
  }
  terminated_ = false;
  game_index_ = 0;
  return true;
}

void DuplicateEnv::Step(int uid) {
  auto state = states_.at(game_index_).get();
  const auto move = game_.GetMove(uid);
  state->ApplyMove(move);
  if (state->IsTerminal() || (!options_.playing_phase &&
                              state->CurrentPhase() > ble::Phase::kAuction)) {
    ++game_index_;
    if (game_index_ > 1) {
      terminated_ = true;
    }
  }
}

float DuplicateEnv::PlayerReward(int player) const {
  const int side = (player & 1);
  const int score1 = options_.playing_phase
                         ? states_[0]->Scores()[side]
                         : states_[0]->ScoreForContracts(
                               side, {states_[0]->GetContract().Index()})[0];
  const int score2 = options_.playing_phase
                         ? states_[1]->Scores()[side]
                         : states_[1]->ScoreForContracts(
                               side, {states_[1]->GetContract().Index()})[0];
  const float reward = static_cast<float>(ble::GetImp(score1, score2)) / 24.0f;
  return reward;
}

std::vector<float> DuplicateEnv::Rewards() const {
  const float team1_reward = PlayerReward(0);
  return {team1_reward, -team1_reward, team1_reward, -team1_reward};
}

std::string DuplicateEnv::ToString() const {
  const std::string rv =
      absl::StrCat("Open table:\n", states_[0]->ToString(), "\nClose table:\n",
                   states_[1]->ToString());
  return rv;
}

std::vector<int> DuplicateEnv::LegalActions() const {
  auto state = states_.at(game_index_).get();
  const auto legal_moves = state->LegalMoves();
  std::vector<int> legal_actions;
  legal_actions.reserve(legal_moves.size());
  for (const auto& move : legal_moves) {
    const int uid = game_.GetMoveUid(move);
    legal_actions.push_back(uid);
  }
  return legal_actions;
}

rela::TensorDict DuplicateEnv::Feature() const {
  rela::TensorDict feature = {};
  auto state = states_.at(game_index_).get();
  const auto observation = ble::BridgeObservation(*state);
  const auto encoding = encoder_.Encode(observation);
  const auto& legal_moves = observation.LegalMoves();
  std::vector<float> legal_move_mask(game_.NumDistinctActions(), 0);
  for (const auto& move : legal_moves) {
    const int uid = game_.GetMoveUid(move);
    legal_move_mask[uid] = 1;
  }
  feature["s"] = torch::tensor(encoding, {torch::kFloat32});
  feature["legal_move"] = torch::tensor(legal_move_mask, {torch::kFloat32});

  // std::cout << "phase: " << static_cast<int>(state->CurrentPhase()) << std::endl;
  if (state->CurrentPhase() == ble::Phase::kAuction) {
    // std::cout << "Options: " << options_.pbe_feature << ", " << options_.jps_feature << ", " << options_.dnns_feature << std::endl;
    if (options_.pbe_feature) {
      // Add pbe feature with key "pbe_s"
      const std::vector<int> pbe_feature = pbe_encoder_.Encode({*state});
      feature["pbe_s"] = torch::tensor(
          std::vector<int>(pbe_feature.begin(), pbe_feature.begin() + 94),
          {torch::kFloat32});
      int convert = pbe_feature.back();
      feature["pbe_convert"] = torch::tensor({convert});
    }
    if (options_.jps_feature) {
      // Add jps feature with key "jps_s", "jps_legal_move"
      const std::vector<int> jps_feature = jps_encoder_.Encode({*state});
      feature["jps_s"] = torch::tensor(jps_feature, {torch::kFloat32});
      const std::vector<int> jps_legal_move(jps_feature.end() - 39,
                                            jps_feature.end());
      feature["jps_legal_move"] =
          torch::tensor(jps_legal_move, {torch::kFloat32});
    }
    if (options_.dnns_feature) {
      // Add jps feature with key "dnns_s"
      const std::vector<int> dnns_feature = dnns_encoder_.Encode({*state});
      feature["dnns_s"] = torch::tensor(dnns_feature, {torch::kFloat32});
    }
  }

  return feature;
}
}  // namespace rlcc