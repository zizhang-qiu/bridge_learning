#ifndef BRIDGE_GAME
#define BRIDGE_GAME
#include "utils.h"
#include "bridge_utils.h"
namespace bridge{
    class BridgeGame{
        public:

        explicit BridgeGame(const Parameters params);
        int NumDistinctActions() const{return kNumCards + kNumCalls;}
        int MaxChanceOutcomes() const{return kNumCards;}
        int MaxUtility() const{return kMaxUtility;}
        int MinUtility() const{return kMinUtility;}

        private:
        Parameters params_;
        bool IsDealerVulnerable() const;
        bool IsNonDealerVulnerable() const;
    };
}

#endif /* BRIDGE_GAME */
