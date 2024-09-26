//
// Created by 13738 on 2024/8/8.
//

#ifndef BRIDGE_LEARNING_RLCC_DETAILED_ENCODER_H_
#define BRIDGE_LEARNING_RLCC_DETAILED_ENCODER_H_
#include "bridge_lib/observation_encoder.h"
#include "bridge_lib/canonical_encoder.h"
namespace ble = bridge_learning_env;


int EncodeAuctionDetailed(const ble::BridgeObservation &obs, int start_offset, std::vector<int> *encoding);
int EncoderTurn(const ble::BridgeObservation &obs, int start_offset, std::vector<int> *encoding);

class DetailedEncoder final: public ble::ObservationEncoder {
 public:
  DetailedEncoder(const std::shared_ptr<ble::BridgeGame> &game,
                  bool turn) : parent_game_(game), turn_(turn) {}
  std::vector<int> Shape() const override;
  std::vector<int> Encode(const ble::BridgeObservation &obs,
                          const std::unordered_map<std::string, std::any> &kwargs) const override;
  Type type() const override { return kDetailed; }
  ObservationEncoder::EncoderPhase EncodingPhase() const override {
    return kAuction;
  }
 private:

  std::shared_ptr<ble::BridgeGame> parent_game_;
  bool turn_;
};
#endif //BRIDGE_LEARNING_RLCC_DETAILED_ENCODER_H_
