//
// Created by qzz on 2023/10/5.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_OBSERVATION_ENCODER_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_OBSERVATION_ENCODER_H_
#include <vector>
#include <map>

#include "bridge_observation.h"

namespace bridge_learning_env {

class ObservationEncoder {
 public:
  enum Type { kCanonical = 0, kPBE = 1, kJPS = 2, kDNNS = 3, kDetailed = 4 };
  virtual ~ObservationEncoder() = default;

  // Returns the shape (dimension sizes of the tensor).
  virtual std::vector<int> Shape() const = 0;

  // All of the canonical observation encodings are vectors of bits. We can
  // change this if we want something more general (e.g. floats or doubles).
  virtual std::vector<int> Encode(const BridgeObservation &obs) const = 0;

  // Return the type of this encoder.
  virtual Type type() const = 0;
 private:

};

class ObservationEncoderFactory {
 public:
  virtual ~ObservationEncoderFactory() = default;

  virtual std::unique_ptr<ObservationEncoder> Create(
      std::shared_ptr<const BridgeGame> game,
      const GameParameters &encoder_params
  ) = 0;
};

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define REGISTER_OBSERVATION_ENCODER(info, factory)  ObservationEncoderRegisterer CONCAT(bot, __COUNTER__)(info, std::make_unique<factory>());

class ObservationEncoderRegisterer {
 public:
  ObservationEncoderRegisterer(const ObservationEncoder::Type type,
                               std::unique_ptr<ObservationEncoderFactory> factory){
    RegisterEncoder(type, std::move(factory));
  }

  static std::unique_ptr<ObservationEncoder> CreateByType(const ObservationEncoder::Type type,
                                                          std::shared_ptr<const BridgeGame> game,
                                                          const GameParameters &encoder_params) {
    auto iter = factories().find(type);
    if (iter == factories().end()) {
      std::cout << "Unknown encoder :" <<
                                  type,
                                   "'. Available encoders are:\n";
      for(const auto t : RegisteredEncoders()){
        std::cout << t << ", ";
      }
      std::cout << std::endl;
      std::abort();
    }else{
      const std::unique_ptr<ObservationEncoderFactory>& factory = iter->second;
      return factory->Create(std::move(game), encoder_params);
    }
  }

  static std::vector<ObservationEncoder::Type> RegisteredEncoders() {
    std::vector<ObservationEncoder::Type> types;
    for (const auto &kv : factories()) types.push_back(kv.first);
    return types;
  }

  static bool IsEncoderRegistered(const ObservationEncoder::Type type) {
    return factories().count(type);
  }

  static void RegisterEncoder(const ObservationEncoder::Type type,
                              std::unique_ptr<ObservationEncoderFactory> factory) {
    factories()[type] = std::move(factory);
  }
 private:
  static std::map<ObservationEncoder::Type, std::unique_ptr<ObservationEncoderFactory>> &factories() {
    static std::map<ObservationEncoder::Type, std::unique_ptr<ObservationEncoderFactory>> impl;
    return impl;
  }
};

} // bridge

#endif //BRIDGE_LEARNING_BRIDGE_LIB_OBSERVATION_ENCODER_H_
