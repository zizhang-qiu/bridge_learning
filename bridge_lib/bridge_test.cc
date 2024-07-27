//
// Created by qzz on 2023/11/13.
//
#include <algorithm>
#include <chrono>

#include "bridge_scoring.h"
#include "bridge_state.h"
#include "example_cards_ddts.h"
#include "utils.h"

namespace bridge_learning_env {
void ContractMadeNotVulTest() {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kUndoubled};
    const Contract heart_contract{level, kHeartsTrump, kUndoubled};
    const int major_contract_points = 30 * level;
    // whether it's small slam or grand slam + part score contract or game bid
    const int major_bonus = std::max(0, level - 5) * 500 +
                            (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kUndoubled};
    const Contract club_contract{level, kClubsTrump, kUndoubled};
    const int minor_contract_points = 20 * level;
    const int minor_bonus = std::max(0, level - 5) * 500 +
                            (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kUndoubled};
    const int no_trump_contract_points = 30 * level + 10;
    const int no_trump_bonus = std::max(0, level - 5) * 500 +
                               (no_trump_contract_points >= 100 ? 300 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 30 * num_over_tricks;
      const int minor_over_trick_points = 20 * num_over_tricks;
      const int no_trump_over_trick_points = 30 * num_over_tricks;
      REQUIRE_EQ(Score(spade_contract, tricks, false),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(heart_contract, tricks, false),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(diamond_contract, tricks, false),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(club_contract, tricks, false),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(no_trump_contract, tricks, false),
                 no_trump_contract_points + no_trump_over_trick_points +
                     no_trump_bonus);
    }
  }

  // doubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kDoubled};
    const Contract heart_contract{level, kHeartsTrump, kDoubled};
    const int major_contract_points = 60 * level;
    // doubled contract always get a 50 bonus
    const int major_bonus = 50 + std::max(0, level - 5) * 500 +
                            (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kDoubled};
    const Contract club_contract{level, kClubsTrump, kDoubled};
    const int minor_contract_points = 40 * level;
    const int minor_bonus = 50 + std::max(0, level - 5) * 500 +
                            (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kDoubled};
    const int no_trump_contract_points = 60 * level + 20;
    const int no_trump_bonus = 50 + std::max(0, level - 5) * 500 +
                               (no_trump_contract_points >= 100 ? 300 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 100 * num_over_tricks;
      const int minor_over_trick_points = 100 * num_over_tricks;
      const int no_trump_over_trick_points = 100 * num_over_tricks;
      REQUIRE_EQ(Score(spade_contract, tricks, false),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(heart_contract, tricks, false),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(diamond_contract, tricks, false),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(club_contract, tricks, false),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(no_trump_contract, tricks, false),
                 no_trump_contract_points + no_trump_over_trick_points +
                     no_trump_bonus);
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kRedoubled};
    const Contract heart_contract{level, kHeartsTrump, kRedoubled};
    const int major_contract_points = 120 * level;
    // redoubled contract always get a 100 bonus
    const int major_bonus = 100 + std::max(0, level - 5) * 500 +
                            (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kRedoubled};
    const Contract club_contract{level, kClubsTrump, kRedoubled};
    const int minor_contract_points = 80 * level;
    const int minor_bonus = 100 + std::max(0, level - 5) * 500 +
                            (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kRedoubled};
    const int no_trump_contract_points = 120 * level + 40;
    const int no_trump_bonus = 100 + std::max(0, level - 5) * 500 +
                               (no_trump_contract_points >= 100 ? 300 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 200 * num_over_tricks;
      const int minor_over_trick_points = 200 * num_over_tricks;
      const int no_trump_over_trick_points = 200 * num_over_tricks;
      REQUIRE_EQ(Score(spade_contract, tricks, false),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(heart_contract, tricks, false),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(diamond_contract, tricks, false),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(club_contract, tricks, false),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(no_trump_contract, tricks, false),
                 no_trump_contract_points + no_trump_over_trick_points +
                     no_trump_bonus);
    }
  }
}

void ContractMadeVulTest() {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kUndoubled};
    const Contract heart_contract{level, kHeartsTrump, kUndoubled};
    const int major_contract_points = 30 * level;
    // whether it's small slam or grand slam + part score contract or game bid
    const int major_bonus = std::max(0, level - 5) * 750 +
                            (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kUndoubled};
    const Contract club_contract{level, kClubsTrump, kUndoubled};
    const int minor_contract_points = 20 * level;
    const int minor_bonus = std::max(0, level - 5) * 750 +
                            (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kUndoubled};
    const int no_trump_contract_points = 30 * level + 10;
    const int no_trump_bonus = std::max(0, level - 5) * 750 +
                               (no_trump_contract_points >= 100 ? 500 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 30 * num_over_tricks;
      const int minor_over_trick_points = 20 * num_over_tricks;
      const int no_trump_over_trick_points = 30 * num_over_tricks;
      REQUIRE_EQ(Score(spade_contract, tricks, true),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(heart_contract, tricks, true),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(diamond_contract, tricks, true),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(club_contract, tricks, true),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(no_trump_contract, tricks, true),
                 no_trump_contract_points + no_trump_over_trick_points +
                     no_trump_bonus);
    }
  }

  // doubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kDoubled};
    const Contract heart_contract{level, kHeartsTrump, kDoubled};
    const int major_contract_points = 60 * level;
    // doubled contract always get a 50 bonus
    const int major_bonus = 50 + std::max(0, level - 5) * 750 +
                            (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kDoubled};
    const Contract club_contract{level, kClubsTrump, kDoubled};
    const int minor_contract_points = 40 * level;
    const int minor_bonus = 50 + std::max(0, level - 5) * 750 +
                            (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kDoubled};
    const int no_trump_contract_points = 60 * level + 20;
    const int no_trump_bonus = 50 + std::max(0, level - 5) * 750 +
                               (no_trump_contract_points >= 100 ? 500 : 50);

    for (int tricks = level + 6; tricks <= kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 200 * num_over_tricks;
      const int minor_over_trick_points = 200 * num_over_tricks;
      const int no_trump_over_trick_points = 200 * num_over_tricks;
      REQUIRE_EQ(Score(spade_contract, tricks, true),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(heart_contract, tricks, true),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(diamond_contract, tricks, true),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(club_contract, tricks, true),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(no_trump_contract, tricks, true),
                 no_trump_contract_points + no_trump_over_trick_points +
                     no_trump_bonus);
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kRedoubled};
    const Contract heart_contract{level, kHeartsTrump, kRedoubled};
    const int major_contract_points = 120 * level;
    // redoubled contract always get a 100 bonus
    const int major_bonus = 100 + std::max(0, level - 5) * 750 +
                            (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kRedoubled};
    const Contract club_contract{level, kClubsTrump, kRedoubled};
    const int minor_contract_points = 80 * level;
    const int minor_bonus = 100 + std::max(0, level - 5) * 750 +
                            (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kRedoubled};
    const int no_trump_contract_points = 120 * level + 40;
    const int no_trump_bonus = 100 + std::max(0, level - 5) * 750 +
                               (no_trump_contract_points >= 100 ? 500 : 50);

    for (int tricks = level + 6;
         tricks <= bridge_learning_env::kNumCardsPerSuit; ++tricks) {
      const int num_over_tricks = tricks - level - 6;
      const int major_over_trick_points = 400 * num_over_tricks;
      const int minor_over_trick_points = 400 * num_over_tricks;
      const int no_trump_over_trick_points = 400 * num_over_tricks;
      REQUIRE_EQ(Score(spade_contract, tricks, true),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(heart_contract, tricks, true),
                 major_contract_points + major_over_trick_points + major_bonus);
      REQUIRE_EQ(Score(diamond_contract, tricks, true),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(club_contract, tricks, true),
                 minor_contract_points + minor_over_trick_points + minor_bonus);
      REQUIRE_EQ(Score(no_trump_contract, tricks, true),
                 no_trump_contract_points + no_trump_over_trick_points +
                     no_trump_bonus);
    }
  }
}

void ContractDefeatedNotVulTest() {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump :
         {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kUndoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = 50 * num_under_tricks;
        REQUIRE_EQ(Score(contract, tricks, false), -penalty);
      }
    }
  }

  // doubled
  constexpr int kPenalty[] = {0, 100, 300, 500, 800};
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump :
         {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kDoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4)
                                ? kPenalty[num_under_tricks]
                                : 300 * (num_under_tricks - 4) + 800;
        REQUIRE_EQ(Score(contract, tricks, false), -penalty);
      }
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump :
         {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kRedoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4)
                                ? (kPenalty[num_under_tricks] * 2)
                                : 600 * (num_under_tricks - 4) + 1600;
        REQUIRE_EQ(Score(contract, tricks, false), -penalty);
      }
    }
  }
}

void ContractDefeatedVulTest() {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump :
         {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kUndoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = 100 * num_under_tricks;
        REQUIRE_EQ(Score(contract, tricks, true), -penalty);
      }
    }
  }

  // doubled
  constexpr int kPenalty[] = {0, 200, 500, 800, 1100};
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump :
         {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kDoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4)
                                ? kPenalty[num_under_tricks]
                                : 300 * (num_under_tricks - 4) + 1100;
        REQUIRE_EQ(Score(contract, tricks, true), -penalty);
      }
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump :
         {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kRedoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4)
                                ? (kPenalty[num_under_tricks] * 2)
                                : 600 * (num_under_tricks - 4) + 2200;
        REQUIRE_EQ(Score(contract, tricks, true), -penalty);
      }
    }
  }
}

void DoubleDummyTest() {
  const size_t num_examples = example_deals.size();
  for (size_t i = 0; i < num_examples; ++i) {
    const auto& deal = example_deals[i];
    const auto& expected_ddt = example_ddts[i];
    auto state = BridgeState(default_game);
    for (const int uid : deal) {
      const BridgeMove move = default_game->GetChanceOutcome(uid);
      state.ApplyMove(move);
    }
    const auto ddt = state.DoubleDummyResults();
    std::vector<int> flattened_ddt(kNumPlayers * kNumDenominations);
    int index = 0;
    for (const auto trump_res : ddt) {
      for (const int res : trump_res) {
        flattened_ddt[index++] = res;
      }
    }
    REQUIRE_VECTOR_EQ(flattened_ddt, std::vector<int>(expected_ddt.begin(), expected_ddt.end()));
    // std::cout << (flattened_ddt == expected_ddt) << std::endl;
  }
}

template <typename T>
T UniformSample(const std::vector<T>& vec, std::mt19937& rng) {
  // Check if the vector is not empty
  if (vec.empty()) {
    throw std::out_of_range("Vector is empty");
  }

  if (vec.size() == 1) {
    return vec[0];
  }

  // Use the provided RNG to generate an index uniformly
  std::uniform_int_distribution<std::size_t> dist(0, vec.size() - 1);

  // Return the sampled element
  return vec[dist(rng)];
}

void RandomSimTest(int num_games, int game_seed, int sample_seed){
  std::cout << "Starting random sim test." << std::endl;
  std::mt19937 rng(sample_seed);
  const GameParameters params = {{"seed", std::to_string(game_seed)}};
  const auto game = std::make_shared<BridgeGame>(params);
  auto st = std::chrono::high_resolution_clock::now();
  for(int i_sim=0; i_sim<num_games; ++i_sim){

    auto state = BridgeState(game);
    while(!state.IsTerminal()){
      if (state.IsChanceNode()){
        state.ApplyRandomChance();
        continue;
      }
      const auto legal_moves = state.LegalMoves();
      const auto move = UniformSample(legal_moves, rng);
      state.ApplyMove(move);
    }
  }
  auto ed = std::chrono::high_resolution_clock::now();
  auto elapsed = ed - st;
  std::cout << "Passed random sim test." << std::endl;
  std::cout << num_games << " games have been simulated, "
            << "Avg time for a game: "
            << double(std::chrono::duration_cast<std::chrono::milliseconds>(
                          elapsed)
                          .count()) /
                   (num_games * 1000)
            << " seconds." << std::endl;
}

}  // namespace bridge_learning_env

int main() {
  bridge_learning_env::ContractMadeNotVulTest();
  bridge_learning_env::ContractMadeVulTest();
  bridge_learning_env::ContractDefeatedNotVulTest();
  bridge_learning_env::ContractDefeatedVulTest();
  std::cout << "Score test passed." << std::endl;
  bridge_learning_env::DoubleDummyTest();
  std::cout << "DD test passed." << std::endl;
  bridge_learning_env::RandomSimTest(10000, 42, 1);
  return 0;
}
