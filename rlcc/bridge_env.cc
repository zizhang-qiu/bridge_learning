//
// Created by qzz on 2023/10/7.
//

#include "bridge_env.h"
using Phase = ble::Phase;

namespace rlcc {

BridgeEnv::BridgeEnv(const ble::GameParameters& params, const bool verbose)
    : params_(params), game_(params_), state_(nullptr), verbose_(verbose),
      encoder_(std::make_shared<ble::BridgeGame>(game_)),
      last_active_player_(ble::kChancePlayerId),
      last_move_(ble::BridgeMove::kInvalid,
                 /*suit=*/ble::kInvalidSuit,
                 /*rank=*/-1,
                 /*denomination=*/ble::kInvalidDenomination,
                 /*level=*/-1,
                 /*other_call=*/ble::kNotOtherCall) {
  if (verbose_) {
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
  return state_->CurrentPhase() > Phase::kAuction;
}

void BridgeEnv::Reset() {
  RELA_CHECK(Terminated())
  state_ = std::make_unique<ble::BridgeState>(std::make_shared<ble::BridgeGame>(game_));
  while (state_->CurrentPhase() == ble::Phase::kDeal) {
    state_->ApplyRandomChance();
  }
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
  int contract_index = state_->GetContract().Index();
  int north_score = state_->ScoreForContracts(ble::Seat::kNorth, {contract_index})[0];
  return {north_score, -north_score, north_score, -north_score};
}

void BridgeEnv::ResetWithDeck(const std::vector<int> &cards) {
  RELA_CHECK_EQ(cards.size(), ble::kNumCards);
  state_ = std::make_unique<ble::BridgeState>(std::make_shared<bridge_learning_env::BridgeGame>(game_));
  for (const int card : cards) {
    const ble::BridgeMove move = game_.GetChanceOutcome(card);
    state_->ApplyMove(move);
  }
}
void BridgeEnv::Step(const ble::BridgeMove& move) {
  RELA_CHECK(!Terminated())
  last_active_player_ = state_->CurrentPlayer();
  last_move_ = move;
  state_->ApplyMove(move);
}

void BridgeEnv::Step(const int uid) {
  const auto move = GetMove(uid);
  Step(move);
}

ble::Player BridgeEnv::CurrentPlayer() const {
  RELA_CHECK_NOTNULL(state_);
  return state_->CurrentPlayer();
}
rela::TensorDict BridgeEnv::Feature() const {
  RELA_CHECK_NOTNULL(state_);
  if (Terminated()) {
    return TerminalFeature();
  }
  const auto observation = ble::BridgeObservation(*state_, state_->CurrentPlayer());
  const auto encoding = encoder_.Encode(observation);
  const auto& legal_moves = observation.LegalMoves();
  std::vector<float> legal_move_mask(ble::kNumCalls, 0);
  for (const auto &move : legal_moves) {
    const int uid = game_.GetMoveUid(move);
    legal_move_mask[uid - ble::kBiddingActionBase] = 1;
  }
  rela::TensorDict res = {
      {"s", torch::tensor(encoding, {torch::kFloat32})},
      {"legal_move", torch::tensor(legal_move_mask, {torch::kFloat32})}
  };
  return res;
}
void BridgeEnv::ResetWithDeckAndDoubleDummyResults(const vector<int> &cards, const vector<int> &double_dummy_results) {
  RELA_CHECK_EQ(cards.size(), ble::kNumCards);
  state_ = std::make_unique<ble::BridgeState>(std::make_shared<bridge_learning_env::BridgeGame>(game_));
  for (const int card : cards) {
    const ble::BridgeMove move = game_.GetChanceOutcome(card);
    state_->ApplyMove(move);
  }
  state_->SetDoubleDummyResults(double_dummy_results);
}

void BridgeEnv::ResetWithBridgeData() {
  RELA_CHECK_NOTNULL(bridge_dataset_);
  const BridgeData bridge_data = bridge_dataset_->Next();
  state_ = std::make_unique<ble::BridgeState>(std::make_shared<bridge_learning_env::BridgeGame>(game_));
  for (const int card : bridge_data.deal) {
    const ble::BridgeMove move = game_.GetChanceOutcome(card);
    state_->ApplyMove(move);
  }
  if (bridge_data.ddt.has_value()) {
    state_->SetDoubleDummyResults(bridge_data.ddt.value());
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
  rela::TensorDict feature = {
      {"s", torch::zeros(encoder_.Shape()[0], {torch::kFloat32})},
      {"legal_move", torch::ones(ble::kNumCalls, {torch::kFloat32})}
  };
  return feature;
}

void BridgeVecEnv::Reset() {
  for (auto &env : envs_) {
    if (env->Terminated()) {
      env->ResetWithBridgeData();
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
  num_envs = min(num_envs, Size());
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
    obs_vec.push_back(env->Feature());
  }
  auto feature = rela::tensor_dict::stack(obs_vec, 0);
  return feature;
}
}

