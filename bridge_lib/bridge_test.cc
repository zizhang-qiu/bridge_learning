//
// Created by qzz on 2023/11/13.
//
#include <algorithm>

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
    const int major_bonus = max(0, level - 5) * 500 + (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kUndoubled};
    const Contract club_contract{level, kClubsTrump, kUndoubled};
    const int minor_contract_points = 20 * level;
    const int minor_bonus = max(0, level - 5) * 500 + (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kUndoubled};
    const int no_trump_contract_points = 30 * level + 10;
    const int no_trump_bonus = max(0, level - 5) * 500 + (no_trump_contract_points >= 100 ? 300 : 50);

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
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // doubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kDoubled};
    const Contract heart_contract{level, kHeartsTrump, kDoubled};
    const int major_contract_points = 60 * level;
    // doubled contract always get a 50 bonus
    const int major_bonus = 50 + max(0, level - 5) * 500 + (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kDoubled};
    const Contract club_contract{level, kClubsTrump, kDoubled};
    const int minor_contract_points = 40 * level;
    const int minor_bonus = 50 + max(0, level - 5) * 500 + (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kDoubled};
    const int no_trump_contract_points = 60 * level + 20;
    const int no_trump_bonus = 50 + max(0, level - 5) * 500 + (no_trump_contract_points >= 100 ? 300 : 50);

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
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kRedoubled};
    const Contract heart_contract{level, kHeartsTrump, kRedoubled};
    const int major_contract_points = 120 * level;
    // redoubled contract always get a 100 bonus
    const int major_bonus = 100 + max(0, level - 5) * 500 + (major_contract_points >= 100 ? 300 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kRedoubled};
    const Contract club_contract{level, kClubsTrump, kRedoubled};
    const int minor_contract_points = 80 * level;
    const int minor_bonus = 100 + max(0, level - 5) * 500 + (minor_contract_points >= 100 ? 300 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kRedoubled};
    const int no_trump_contract_points = 120 * level + 40;
    const int no_trump_bonus = 100 + max(0, level - 5) * 500 + (no_trump_contract_points >= 100 ? 300 : 50);

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
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }
}

void ContractMadeVulTest(){
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kUndoubled};
    const Contract heart_contract{level, kHeartsTrump, kUndoubled};
    const int major_contract_points = 30 * level;
    // whether it's small slam or grand slam + part score contract or game bid
    const int major_bonus = max(0, level - 5) * 750 + (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kUndoubled};
    const Contract club_contract{level, kClubsTrump, kUndoubled};
    const int minor_contract_points = 20 * level;
    const int minor_bonus = max(0, level - 5) * 750 + (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kUndoubled};
    const int no_trump_contract_points = 30 * level + 10;
    const int no_trump_bonus = max(0, level - 5) * 750 + (no_trump_contract_points >= 100 ? 500 : 50);

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
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // doubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kDoubled};
    const Contract heart_contract{level, kHeartsTrump, kDoubled};
    const int major_contract_points = 60 * level;
    // doubled contract always get a 50 bonus
    const int major_bonus = 50 + max(0, level - 5) * 750 + (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kDoubled};
    const Contract club_contract{level, kClubsTrump, kDoubled};
    const int minor_contract_points = 40 * level;
    const int minor_bonus = 50 + max(0, level - 5) * 750 + (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kDoubled};
    const int no_trump_contract_points = 60 * level + 20;
    const int no_trump_bonus = 50 + max(0, level - 5) * 750 + (no_trump_contract_points >= 100 ? 500 : 50);

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
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    // major suits
    const Contract spade_contract{level, kSpadesTrump, kRedoubled};
    const Contract heart_contract{level, kHeartsTrump, kRedoubled};
    const int major_contract_points = 120 * level;
    // redoubled contract always get a 100 bonus
    const int major_bonus = 100 + max(0, level - 5) * 750 + (major_contract_points >= 100 ? 500 : 50);

    // minor suits
    const Contract diamond_contract{level, kDiamondsTrump, kRedoubled};
    const Contract club_contract{level, kClubsTrump, kRedoubled};
    const int minor_contract_points = 80 * level;
    const int minor_bonus = 100 + max(0, level - 5) * 750 + (minor_contract_points >= 100 ? 500 : 50);

    // no trump
    const Contract no_trump_contract{level, kNoTrump, kRedoubled};
    const int no_trump_contract_points = 120 * level + 40;
    const int no_trump_bonus = 100 + max(0, level - 5) * 750 + (no_trump_contract_points >= 100 ? 500 : 50);

    for (int tricks = level + 6; tricks <= bridge_learning_env::kNumCardsPerSuit; ++tricks) {
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
                no_trump_contract_points + no_trump_over_trick_points + no_trump_bonus);
    }
  }
}

void ContractDefeatedNotVulTest(){
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
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
    for (const auto trump : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kDoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4) ? kPenalty[num_under_tricks] : 300 * (num_under_tricks - 4) + 800;
        REQUIRE_EQ(Score(contract, tricks, false), -penalty);
      }
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kRedoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty =
            (num_under_tricks <= 4) ? (kPenalty[num_under_tricks] * 2) : 600 * (num_under_tricks - 4) + 1600;
        REQUIRE_EQ(Score(contract, tricks, false), -penalty);
      }
    }
  }
}

void ContractDefeatedVulTest() {
  // undoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
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
    for (const auto trump : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kDoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty = (num_under_tricks <= 4) ? kPenalty[num_under_tricks] : 300 * (num_under_tricks - 4) + 1100;
        REQUIRE_EQ(Score(contract, tricks, true), -penalty);
      }
    }
  }

  // redoubled
  for (int level = 1; level <= 7; ++level) {
    for (const auto trump : {kClubsTrump, kDiamondsTrump, kHeartsTrump, kSpadesTrump, kNoTrump}) {
      const Contract contract{level, trump, kRedoubled};
      for (int tricks = 0; tricks < level + 6; ++tricks) {
        const int num_under_tricks = level + 6 - tricks;
        const int penalty =
            (num_under_tricks <= 4) ? (kPenalty[num_under_tricks] * 2) : 600 * (num_under_tricks - 4) + 2200;
        REQUIRE_EQ(Score(contract, tricks, true), -penalty);
      }
    }
  }
}

void DoubleDummyTest() {
  const size_t num_examples = example_deals.size();
  for(size_t i=0; i<num_examples; ++i) {
    const auto& deal = example_deals[i];
    const auto& expected_ddt = example_ddts[i];
    auto state = BridgeState(default_game);
    for(const int uid:deal) {
      const BridgeMove move = default_game->GetChanceOutcome(uid);
      state.ApplyMove(move);
    }
    const auto ddt = state.DoubleDummyResults();
    std::vector<int> flattened_ddt(kNumPlayers * kNumDenominations);
    int index = 0;
    for(const auto trump_res:ddt) {
      for(const int res:trump_res) {
        flattened_ddt[index++] = res;
      }
    }
    REQUIRE_VECTOR_EQ(flattened_ddt, expected_ddt);
    // std::cout << (flattened_ddt == expected_ddt) << std::endl;
  }
}
}






int main() {
  bridge_learning_env::ContractMadeNotVulTest();
  bridge_learning_env::ContractMadeVulTest();
  bridge_learning_env::ContractDefeatedNotVulTest();
  bridge_learning_env::ContractDefeatedVulTest();
  bridge_learning_env::DoubleDummyTest();
}
