//
// Created by qzz on 2023/10/5.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_OBSERVATION_ENCODER_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_OBSERVATION_ENCODER_H_
#include <utility>
#include <vector>
#include <map>
#include <any>

#include "bridge_observation.h"

namespace bridge_learning_env {

class ObservationEncoder {
 public:
  enum Type { kCanonical = 0, kPBE = 1, kJPS = 2, kDNNS = 3, kDetailed = 4 };
  enum EncoderPhase { kUnknown = -1, kAuction = 0, kPlay = 1, kBoth = 2 };
  virtual ~ObservationEncoder() = default;

  // Returns the shape (dimension sizes of the tensor).
  virtual std::vector<int> Shape() const = 0;

  // All of the canonical observation encodings are vectors of bits. We can
  // change this if we want something more general (e.g. floats or doubles).
  virtual std::vector<int> Encode(const BridgeObservation &obs,
                                  const std::unordered_map<std::string, std::any> &kwargs = {}) const = 0;

  virtual EncoderPhase EncodingPhase() const { return kUnknown; }

  // Return the type of this encoder.
  virtual Type type() const = 0;
 private:

};

} // bridge

#endif //BRIDGE_LEARNING_BRIDGE_LIB_OBSERVATION_ENCODER_H_
