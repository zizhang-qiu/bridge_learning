//
// Created by qzz on 2023/10/7.
//

#include "bridge_env.h"
#include <random>
#include <vector>
#include "bridge_lib/bridge_utils.h"
#include "rela/logging.h"
#include "rela/utils.h"
using Phase = ble::Phase;

namespace rlcc {

void WarnOnce(const std::string &warning_msg) {
  static bool is_first_call = true;
  if (is_first_call) {
    std::cout << warning_msg << std::endl;
    is_first_call = false;
  }
}

template<typename Container>
ble::BridgeState StateFromTrajectory(
    const Container &trajectory, const std::shared_ptr<ble::BridgeGame> &game) {
  RELA_CHECK_GE(trajectory.size(), ble::kNumCards);
  ble::BridgeState state{game};
  // Deal.
  for (int i = 0; i < ble::kNumCards; ++i) {
    const auto move = game->GetChanceOutcome(trajectory[i]);
    state.ApplyMove(move);
  }

  // Remained moves.
  for (int i = ble::kNumCards; i < trajectory.size(); ++i) {
    const auto move = game->GetMove(trajectory[i]);
    state.ApplyMove(move);
  }

  return state;
}

BridgeEnv::BridgeEnv(const ble::GameParameters &params,
                     const BridgeEnvOptions &options)
    : params_(params),
      game_(params_),
      options_(options),
      state_(nullptr),
      last_active_player_(ble::kChancePlayerId),
      last_move_(),
      max_len_(options.max_len) {
  if (!options.bidding_phase && !options.playing_phase) {
    rela::utils::RelaFatalError(
        "Both bidding and playing phase are off. At least one phase should be "
        "on.");
  }
  const auto game = std::make_shared<ble::BridgeGame>(game_);
  encoder_ = std::move(rlcc::LoadEncoder(options_.encoder, game));
  if (options_.verbose) {
    std::cout << "Bridge game created, with parameters:\n";
    for (const auto &item : params) {
      std::cout << "  " << item.first << "=" << item.second << "\n";
    }
  }
}

bool BridgeEnv::Terminated() const {
  if (state_ == nullptr) {
    return true;
  }
  if (options_.playing_phase) {
    return state_->IsTerminal();
  }

  return state_->CurrentPhase() > Phase::kAuction || (max_len_ > 0 && num_step_ == max_len_);
}

bool BridgeEnv::Reset() {
  RELA_CHECK(Terminated())
  if (bridge_dataset_ == nullptr) {
    state_ = std::make_unique<ble::BridgeState>(
        std::make_shared<ble::BridgeGame>(game_));
    while (state_->CurrentPhase() == ble::Phase::kDeal) {
      state_->ApplyRandomChance();
    }
    if (!options_.bidding_phase) {
      WarnOnce(
          "Warning: Use only playing phase without a dataset can cause random "
          "state.");
      std::mt19937 rng;
      while (state_->CurrentPhase() == Phase::kAuction) {
        const auto move = rela::utils::UniformSample(state_->LegalMoves(), rng);
        state_->ApplyMove(move);
      }
    }
  } else {
    ResetWithDataSet();
  }
  num_step_ = 0;
  return true;
}

std::string BridgeEnv::ToString() const {
  if (state_ == nullptr) {
    return "Env not reset.";
  }
  return state_->ToString();
}

std::vector<int> BridgeEnv::Returns() const {
  RELA_CHECK(Terminated())
  RELA_CHECK_NOTNULL(state_);
  // Truncated.
  if (state_->CurrentPhase() == ble::Phase::kAuction) {
    return {0, 0, 0, 0};
  }
  if (state_->IsTerminal()){
    return state_->Scores();
  }
  int contract_index = state_->GetContract().Index();
  int north_score = state_->ScoreForContracts(ble::Seat::kNorth, {contract_index})[0];
  return {north_score, -north_score, north_score, -north_score};
}

void BridgeEnv::ResetWithDeck(const std::vector<int> &cards) {
  RELA_CHECK_EQ(cards.size(), ble::kNumCards);
  state_ = std::make_unique<ble::BridgeState>(
      std::make_shared<bridge_learning_env::BridgeGame>(game_));
  for (const int card : cards) {
    const ble::BridgeMove move = game_.GetChanceOutcome(card);
    state_->ApplyMove(move);
  }
}

void BridgeEnv::Step(const ble::BridgeMove &move) {
  RELA_CHECK(!Terminated())
  last_active_player_ = state_->CurrentPlayer();
  last_move_ = move;
  state_->ApplyMove(move);
  ++num_step_;
}

void BridgeEnv::Step(const int uid) {
  const auto move = GetMove(uid);
  Step(move);
}

ble::Player BridgeEnv::CurrentPlayer() const {
  RELA_CHECK_NOTNULL(state_);
  return state_->CurrentPlayer();
}

rela::TensorDict BridgeEnv::Feature(int player) const {
  if (player == -1) {
    player = CurrentPlayer();
  }
  RELA_CHECK_NOTNULL(state_);
  if (Terminated()) {
    std::cout << "Get feature at terminal\n";
    return TerminalFeature();
  }
  const auto observation = ble::BridgeObservation(*state_, player);
  const auto encoding = encoder_->Encode(observation);
  const auto &legal_moves = observation.LegalMoves();
  std::vector<float> legal_move_mask(MaxNumAction(), 0);
  for (const auto &move : legal_moves) {
    const int uid = game_.GetMoveUid(move);
    legal_move_mask[uid] = 1;
  }
  if (player != CurrentPlayer()) {
    legal_move_mask[NoOPUid()] = 1;
  }
  rela::TensorDict res = {
      {"s", torch::tensor(encoding, {torch::kFloat32})},
      {"legal_move", torch::tensor(legal_move_mask, {torch::kFloat32})}};

  return res;
}

void BridgeEnv::ResetWithDeckAndDoubleDummyResults(
    const std::vector<int> &cards,
    const std::vector<int> &double_dummy_results) {
  RELA_CHECK_EQ(cards.size(), ble::kNumCards);
  state_ = std::make_unique<ble::BridgeState>(
      std::make_shared<bridge_learning_env::BridgeGame>(game_));
  for (const int card : cards) {
    const ble::BridgeMove move = game_.GetChanceOutcome(card);
    state_->ApplyMove(move);
  }
  state_->SetDoubleDummyResults(double_dummy_results);
}

void BridgeEnv::ResetWithDataSet() {
  RELA_CHECK_NOTNULL(bridge_dataset_);
  const BridgeData bridge_data = bridge_dataset_->Next();
  state_ = std::make_unique<ble::BridgeState>(
      std::make_shared<bridge_learning_env::BridgeGame>(game_));
  for (int i = 0; i < ble::kNumCards; ++i) {
    const int card = bridge_data.deal[i];
    const ble::BridgeMove move = game_.GetChanceOutcome(card);
    state_->ApplyMove(move);
  }
  if (bridge_data.ddt.has_value()) {
    state_->SetDoubleDummyResults(bridge_data.ddt.value());
  }
  if (!options_.bidding_phase) {
    RELA_CHECK_GE(bridge_data.deal.size(), ble::kNumCards + ble::kNumPlayers);
    for (int i = ble::kNumCards; i < bridge_data.deal.size(); ++i) {
      const auto move = game_.GetMove(bridge_data.deal[i]);
      state_->ApplyMove(move);
    }
  }
}

const ble::GameParameters &BridgeEnv::Parameters() const {
  return params_;
}

ble::BridgeObservation BridgeEnv::BleObservation() const {
  RELA_CHECK_NOTNULL(state_);
  auto observation = ble::BridgeObservation(*state_, state_->CurrentPlayer());
  return observation;
}

rela::TensorDict BridgeEnv::TerminalFeature() const {
  torch::Tensor legal_move = torch::zeros(MaxNumAction(), {torch::kFloat32});
  legal_move[NoOPUid()] = 1;
  rela::TensorDict feature = {
      {"s", torch::zeros(encoder_->Shape()[0], {torch::kFloat32})},
      {"legal_move", legal_move}};
  return feature;
}

void BridgeVecEnv::Reset() {
  for (auto &env : envs_) {
    if (env->Terminated()) {
      env->ResetWithDataSet();
    }
  }
}

bool BridgeVecEnv::AnyTerminated() const {
  for (const auto &env : envs_) {
    if (env->Terminated()) {
      return true;
    }
  }
  return false;
}

bool BridgeVecEnv::AllTerminated() const {
  for (const auto &env : envs_) {
    if (!env->Terminated()) {
      return false;
    }
  }
  return true;
}

void BridgeVecEnv::Step(rela::TensorDict reply) {
  auto actions = reply.at("a");
  RELA_CHECK_EQ(actions.numel(), Size());
  auto accessor = actions.accessor<int, 1>();
  for (int i = 0; i < accessor.size(0); ++i) {
    if (!envs_[i]->Terminated()) {
      envs_[i]->Step(accessor[i]);
    }
  }
}

void BridgeVecEnv::DisPlay(int num_envs) const {
  num_envs = std::min(num_envs, Size());
  std::string rv;
  for (size_t i = 0; i < num_envs; ++i) {
    rv += "Env " + std::to_string(i) + "\n";
    rv += envs_[i]->ToString() + "\n";
  }
  std::cout << rv << std::endl;
}

rela::TensorDict BridgeVecEnv::Feature() const {
  std::vector<rela::TensorDict> obs_vec;
  obs_vec.reserve(Size());
  for (const auto &env : envs_) {
    obs_vec.push_back(env->Feature(-1));
  }
  auto feature = rela::tensor_dict::stack(obs_vec, 0);
  return feature;
}
}  // namespace rlcc
