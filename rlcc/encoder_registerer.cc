//
// Created by 13738 on 2024/8/25.
//
#include "encoder_registerer.h"
#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/pbe_encoder.h"
#include "bridge_lib/jps_encoder.h"
#include "utils.h"

namespace rlcc {

class CanonicalEncoderFactory : public ObservationEncoderFactory {
 public:
  std::unique_ptr<ble::ObservationEncoder> Create(const std::shared_ptr<ble::BridgeGame> &game,
                                                  const ble::GameParameters &encoder_params) override {
    auto encoder = std::make_unique<ble::CanonicalEncoder>(game);
    return encoder;
  }
};

REGISTER_OBSERVATION_ENCODER("canonical", CanonicalEncoderFactory);

class PBEEncoderFactory : public ObservationEncoderFactory {
 public:
  std::unique_ptr<ble::ObservationEncoder> Create(const std::shared_ptr<ble::BridgeGame> &game,
                                                  const bridge_learning_env::GameParameters &encoder_params) override {
    return std::make_unique<ble::PBEEncoder>(game);
  }
};
REGISTER_OBSERVATION_ENCODER("pbe", CanonicalEncoderFactory);

class JPSEncoderFactory : public ObservationEncoderFactory {
 public:
  std::unique_ptr<ble::ObservationEncoder> Create(const std::shared_ptr<ble::BridgeGame> &game,
                                                  const bridge_learning_env::GameParameters &encoder_params) override {
    return std::make_unique<ble::JPSEncoder>(game);
  }
};

REGISTER_OBSERVATION_ENCODER("jps", JPSEncoderFactory);

std::vector<std::string> RegisteredEncoders() {
  return ObservationEncoderRegisterer::RegisteredEncoders();
}
bool IsEncoderRegistered(const std::string &name) {
  return ObservationEncoderRegisterer::IsEncoderRegistered(name);
}
std::unique_ptr<ble::ObservationEncoder> LoadEncoder(const std::string &name,
                                                     const std::shared_ptr<ble::BridgeGame> &game,
                                                     const bridge_learning_env::GameParameters &encoder_params) {
  std::unique_ptr<ble::ObservationEncoder> result =
      ObservationEncoderRegisterer::CreateByName(name, game, encoder_params);
  if (result == nullptr) {
    std::cerr << "Unable to create encoder: " << name << std::endl;
    std::abort();
  }
  return result;
}

std::unique_ptr<ble::ObservationEncoder> rlcc::LoadEncoder(const std::string &name,
                                                           const std::shared_ptr<ble::BridgeGame> &game) {
  const auto params = ParametersFromString(name);

  return LoadEncoder(params.at("name"), game, params);
}
}