//
// Created by qzz on 2023/10/9.
//
#include "bridge_dataset.h"
#include <mutex>

#include "rela/logging.h"

namespace rlcc{
BridgeDataset::BridgeDataset(const std::vector<std::vector<int>> &deals) {
  size_t num_deals = deals.size();
  for (size_t i = 0; i < num_deals; ++i) {
    const BridgeData deal{deals[i]};
    dataset_.push_back(deal);
  }
}
BridgeDataset::BridgeDataset(const std::vector<std::vector<int>> &deals,
                             const std::vector<std::array<int, kDoubleDummyResultSize>> &ddts) {
  RELA_CHECK_EQ(deals.size(), ddts.size());
  size_t num_deals = deals.size();
  for (size_t i = 0; i < num_deals; ++i) {
    const BridgeData deal{deals[i], ddts[i]};
    dataset_.push_back(deal);
  }
}

BridgeData BridgeDataset::Next() {
  std::lock_guard<std::mutex> lk(m_);
  // RELA_CHECK_GE(Size(), 0);
  const BridgeData data = dataset_[index_];
  index_ = (index_ + 1) % Size();
  return data;
}

}
