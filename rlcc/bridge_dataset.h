//
// Created by qzz on 2023/10/9.
//

#ifndef BRIDGE_LEARNING_RLCC_BRIDGE_DATASET_H_
#define BRIDGE_LEARNING_RLCC_BRIDGE_DATASET_H_
#include <vector>
#include <array>
#include <optional>
#include <mutex>

#include "bridge_lib/bridge_utils.h"
namespace ble = bridge_learning_env;
inline constexpr int kDoubleDummyResultSize = ble::kNumPlayers * ble::kNumDenominations;
struct BridgeData {
  std::vector<int> deal{};
  std::optional<std::array<int, kDoubleDummyResultSize>> ddt;
};

class BridgeDataset {
 public:

  explicit BridgeDataset(const std::vector<std::vector<int>> &deals);

  BridgeDataset(const std::vector<std::vector<int>> &deals,
                const std::vector<std::array<int, kDoubleDummyResultSize>> &ddts);

  int Size() const { return static_cast<int>(dataset_.size()); }

  BridgeData Next();

 private:
  std::vector<BridgeData> dataset_;
  std::mutex m_;
  int index_ = 0;
};
#endif //BRIDGE_LEARNING_RLCC_BRIDGE_DATASET_H_
