import hydra
import torch

import common_utils
from net import MLP
from set_path import append_sys_path

append_sys_path()

import bridge
import rela
import bridgelearn
import bridgeplay


def auction_from_str(auction_str: str):
    if auction_str.lower() in ["pass", "p"]:
        return bridge.BridgeMove(bridge.OtherCalls.PASS)
    if auction_str.lower() in ["d", "dbl", "double", "x"]:
        return bridge.BridgeMove(bridge.OtherCalls.DOUBLE)
    if auction_str.lower() in ["r", "rdbl", "redouble", "xx"]:
        return bridge.BridgeMove(bridge.OtherCalls.REDOUBLE)

    common_utils.assert_eq(len(auction_str), 2)

    level = int(auction_str[0])
    assert 1 <= level <= 7
    denom = "CDHSN".find(auction_str[1].upper())
    common_utils.assert_neq(denom, -1)
    return bridge.BridgeMove(level, bridge.Denomination(denom))


@hydra.main("conf", "console_play", version_base="1.2")
def main(args):
    policy_net: MLP = hydra.utils.instantiate(args.net)
    # print(policy_net)
    policy_net.load_state_dict(torch.load(args.policy_weight))
    policy_net.to(args.device)

    state = bridge.BridgeState(bridge.default_game)
    while state.is_chance_node():
        state.apply_random_chance()

    human_player = input("Input your seat (NESW):\n")
    human_player = "NESW".find(human_player)

    encoder = bridge.CanonicalEncoder(bridge.default_game)

    while state.current_phase() == bridge.Phase.AUCTION:
        if state.current_player() != human_player:
            feature = encoder.encode(bridge.BridgeObservation(state))[:480]
            probs = policy_net.forward(
                torch.unsqueeze(torch.tensor(feature, dtype=torch.float32), 0).to(
                    args.device
                )
            )
            uid = (torch.argmax(probs[0]) + 52).item()
            move = bridge.default_game.get_move(int(uid))
        else:
            print("Your turn.")
            print("Your observation:\n")
            print(bridge.BridgeObservation(state))
            auction_str = input("Input your call:\n")
            move = auction_from_str(auction_str=auction_str)
        state.apply_move(move)
    print("Auction end, final state is:\n")
    print(state)
    contract = state.get_contract()
    declarer = contract.declarer
    print(
        f"Double dummy result: declarer win {state.double_dummy_results()[declarer][contract.denomination]} tricks."
    )
    print(
        f"Declarer score: {state.score_for_contracts(declarer, [contract.index()])[0]}"
    )


if __name__ == "__main__":
    main()
