#include <memory>
#include <random>
#include <vector>
#include "bridge_state.h"
#include "bridge_observation.h"
#include "canonical_encoder.h"

constexpr const char* kGameParamArgPrefix = "--config.bridge.";

std::vector<int> SimulateGame(const bridge_learning_env::BridgeGame& game,
                              bool verbose, std::mt19937* rng) {
  bridge_learning_env::BridgeState state(
      std::make_shared<bridge_learning_env::BridgeGame>(game));
  while (!state.IsTerminal()) {
    // Chance node.
    if (state.CurrentPlayer() == bridge_learning_env::kChancePlayerId) {
      auto chance_outcomes = state.ChanceOutcomes();
      std::discrete_distribution<std::mt19937::result_type> dist(
          chance_outcomes.second.begin(), chance_outcomes.second.end());
      auto move = chance_outcomes.first[dist(*rng)];
      if (verbose) {
        std::cout << "Legal chance:";
        for(int i=0; i<chance_outcomes.first.size(); ++i){
            std::cout << " <" << chance_outcomes.first[i].ToString() << ", " << chance_outcomes.second[i] << ">";
        }
        std::cout << "\n";
        std::cout << "Sampled move: " << move.ToString() << "\n\n";
      }
      state.ApplyMove(move);
      continue;
    }

    const auto obs = bridge_learning_env::BridgeObservation(state);
    bridge_learning_env::CanonicalEncoder encoder(
        std::make_shared<bridge_learning_env::BridgeGame>(game));

    const auto legal_moves = state.LegalMoves();
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, legal_moves.size() - 1);
    auto move = legal_moves[dist(*rng)];
    if (verbose) {
      std::cout << "Current player: " << state.CurrentPlayer() << "\n";
      std::cout << state.ToString() << "\n\n";
      std::cout << "Legal moves:";
      for (int i = 0; i < legal_moves.size(); ++i) {
        std::cout << " " << legal_moves[i].ToString();
      }
      std::cout << "\n";
      std::cout << "Sampled move: " << move.ToString() << "\n\n";
    }
    state.ApplyMove(move);
  }
  if(verbose){
    std::cout << "Game done, terminal state:\n" << state.ToString() << "\n\n";
  }

  return state.Scores();
}

std::unordered_map<std::string, std::string> ParseArguments(int argc,
                                                            char** argv) {
  std::unordered_map<std::string, std::string> game_params;
  const auto prefix_len = strlen(kGameParamArgPrefix);
  for (int i = 1; i < argc; ++i) {
    std::string param = argv[i];
    if (param.compare(0, prefix_len, kGameParamArgPrefix) == 0 &&
        param.size() > prefix_len) {
      std::string value;
      param = param.substr(prefix_len, std::string::npos);
      auto value_pos = param.find('=');
      if (value_pos != std::string::npos) {
        value = param.substr(value_pos + 1, std::string::npos);
        param = param.substr(0, value_pos);
      }
      game_params[param] = value;
    }
  }
  return game_params;
}

int main(int argc, char** argv) {
  auto game_params = ParseArguments(argc, argv);
  auto game = bridge_learning_env::BridgeGame(game_params);
  std::mt19937 rng;
  rng.seed(std::random_device()());
  SimulateGame(game, true, &rng);
}