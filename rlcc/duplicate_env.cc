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
  ++num_steps_;
  if (state->IsTerminal() || (!options_.playing_phase &&
      state->CurrentPhase() > ble::Phase::kAuction)
      || (options_.max_len > 0 && num_steps_ == options_.max_len)) {
    ++game_index_;
    num_steps_ = 0;
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
  for (const auto &move : legal_moves) {
    const int uid = game_.GetMoveUid(move);
    legal_actions.push_back(uid);
  }
  return legal_actions;
}

rela::TensorDict DuplicateEnv::Feature(int player) const {
  const auto state = states_.at(game_index_).get();
  int state_player = 0;
  if (player == -1) {
    state_player = state->CurrentPlayer();
  } else {
    if (game_index_ == 1) {
      state_player = (player + 1) % ble::kNumPlayers;
    } else {
      state_player = player;
    }
  }
  const int acting_player = CurrentPlayer();
  rela::TensorDict feature = {};

  const auto observation = ble::BridgeObservation(*state, state_player);
  const auto encoding = encoder_->Encode(observation);
  if (acting_player != player) {
    feature["legal_move"] = torch::zeros(game_.NumDistinctActions() + 1, {torch::kFloat32});
    feature["legal_move"][game_.NumDistinctActions()] = 1;
  } else {
    const auto &legal_moves = observation.LegalMoves();
    std::vector<float> legal_move_mask(game_.NumDistinctActions() + 1, 0);
    for (const auto &move : legal_moves) {
      const int uid = game_.GetMoveUid(move);
      legal_move_mask[uid] = 1;
    }
    feature["legal_move"] = torch::tensor(legal_move_mask, {torch::kFloat32});
  }

  feature["s"] = torch::tensor(encoding, {torch::kFloat32});

  feature["table_idx"] = torch::tensor(game_index_);
  // std::cout << "phase: " << static_cast<int>(state->CurrentPhase()) << std::endl;

  return feature;
}
}  // namespace rlcc