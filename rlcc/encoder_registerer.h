//
// Created by 13738 on 2024/8/25.
//

#ifndef BRIDGE_LEARNING_RLCC_ENCODER_REGISTERER_H_
#define BRIDGE_LEARNING_RLCC_ENCODER_REGISTERER_H_
#include "bridge_lib/observation_encoder.h"
namespace ble = bridge_learning_env;
namespace rlcc{
class ObservationEncoderFactory {
 public:
  virtual ~ObservationEncoderFactory() = default;

  virtual std::unique_ptr<ble::ObservationEncoder> Create(
      std::shared_ptr<ble::BridgeGame>& game,
      const ble::GameParameters &encoder_params
  ) = 0;
};

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define REGISTER_OBSERVATION_ENCODER(info, factory)  ObservationEncoderRegisterer CONCAT(bot, __COUNTER__)(info, std::make_unique<factory>());

class ObservationEncoderRegisterer {
 public:
  ObservationEncoderRegisterer(const std::string &name,
                               std::unique_ptr<ObservationEncoderFactory> factory) {
    RegisterEncoder(name, std::move(factory));
  }

  static std::unique_ptr<ble::ObservationEncoder> CreateByName(const std::string &name,
                                                          std::shared_ptr<ble::BridgeGame>& game,
                                                          const ble::GameParameters &encoder_params) {
    auto iter = factories().find(name);
    if (iter == factories().end()) {
      std::cout << "Unknown encoder :" <<
                name <<
                "'. Available encoders are:\n";
      for (const auto &t : RegisteredEncoders()) {
        std::cout << t << ", ";
      }
      std::cout << std::endl;
      std::abort();
    } else {
      const std::unique_ptr<ObservationEncoderFactory> &factory = iter->second;
      return factory->Create(game, encoder_params);
    }
  }

  static std::vector<std::string> RegisteredEncoders() {
    std::vector<std::string> names;
    for (const auto &kv : factories()) names.push_back(kv.first);
    return names;
  }

  static bool IsEncoderRegistered(const std::string &name) {
    return factories().count(name);
  }

  static void RegisterEncoder(const std::string &name,
                              std::unique_ptr<ObservationEncoderFactory> factory) {
    factories()[name] = std::move(factory);
  }
 private:
  static std::map<std::string, std::unique_ptr<ObservationEncoderFactory>> &factories() {
    static std::map<std::string, std::unique_ptr<ObservationEncoderFactory>> impl;
    return impl;
  }
};

std::vector<std::string> RegisteredEncoders();

bool IsEncoderRegistered(const std::string& name);

std::unique_ptr<ble::ObservationEncoder> LoadEncoder(const std::string& name,
                                                std::shared_ptr<ble::BridgeGame>& game,
                                                const ble::GameParameters &encoder_params);
}

#endif //BRIDGE_LEARNING_RLCC_ENCODER_REGISTERER_H_
