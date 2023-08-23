#include <iostream>
#include <cassert>
#include <memory>
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/utils.h"
#include "bridge_lib/bridge_scoring.h"

int main(int, char**){
    // for(int i=0; i<bridge::kNumCards; ++i){
    //     std::cout << bridge::CardString(i) << "\n";
    // }
    // const bridge::Parameters params = {
    //    {"expect_true", "true"},
    //     {"expect_1", "1"},
    //     {"expect_5.66", "5.66"},
    //     {"expect_yes", "yes"}
    // };
    // assert( bridge::ParameterValue(params, "expect_true", false)==true);
    // assert(bridge::ParameterValue(params, "expect_1", 1) == 1);
    // assert(bridge::ParameterValue(params, "expect_5.66", 0.01) == 5.66);
    // assert(bridge::ParameterValue(params, "expect_yes", "no") == "yes");
    // for(int i=0; i<bridge::kNumContracts; ++i){
    //     std::cout << bridge::kAllContracts[i].ToString() << "\n";
    // }
    auto cards = bridge::Permutation(bridge::kNumCards);
    bridge::PrintVector(cards);


    auto state = std::make_shared<bridge::BridgeState>(false, false);
    for(int i=0; i<bridge::kNumCards; ++i){
        state->ApplyAction(cards[i]);
    }

    for(const int action:{55, 52, 52, 52}){
      bridge::PrintVector(state->LegalActions());
      state->ApplyAction(action);
        
    }

    std::cout << "current phase: " << state->CurrentPhase() << "\n";



    std::cout << state->ToString() << "\n";
}
