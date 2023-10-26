import argparse
from typing import List, Optional, Union

import numpy as np
import torch

from play_bot import BlueChipBridgeBot
from common_utils import assert_eq
from common_utils import tensor_dict_to_device
from wbridge5_client import WBridge5Client, Controller
from create_bridge import create_params, create_bridge_game
from set_path import append_sys_path
from agent import BridgeAgent, DEFAULT_POLICY_CONF, DEFAULT_VALUE_CONF, SimpleAgent

append_sys_path()
import bridge
import bridgelearn

bot_cmd = "D:/wbridge5/Wbridge5.exe Autoconnect {port}"
timeout_secs = 60
num_deals = 10


def controller_factory() -> Controller:
    client = WBridge5Client(bot_cmd, timeout_secs)
    client.start()
    return client


def make_obs_tensor_dict(state: bridge.BridgeState2):
    parent_game = state.parent_game()
    observation = bridge.BridgeObservation(state, state.current_player())
    encoder = bridge.CanonicalEncoder(state.parent_game())
    s = encoder.encode(observation)
    legal_move_mask = torch.zeros(bridge.NUM_CALLS, dtype=torch.float32)
    legal_moves = state.legal_moves()
    for move in legal_moves:
        uid = parent_game.get_move_uid(move)
        legal_move_mask[uid - bridge.NUM_CARDS] = 1
    obs = {
        "s": torch.tensor(s, dtype=torch.float32),
        "legal_move": legal_move_mask
    }
    return obs


def _bid_and_play(state: bridge.BridgeState2, bots: List[BlueChipBridgeBot], agent: SimpleAgent,
                  pimc_bot: bridgelearn.PIMCBot, agent_seats: List[int], device="cuda"):
    while state.current_phase() == bridge.Phase.AUCTION:
        if state.current_player() in agent_seats:
            obs = make_obs_tensor_dict(state)
            obs = tensor_dict_to_device(obs, device)
            action = agent.act(obs)
            state.apply_move(state.parent_game().get_move(action))
        else:
            result = bots[state.current_player()].step(state)
            state.apply_move(state.parent_game().get_move(result))

    while state.current_phase() == bridge.Phase.PLAY:
        # print(state)
        # play phase
        if state.current_player() in agent_seats:
            # print(state.legal_moves())
            search_result = pimc_bot.search(state)
            # print(search_result.moves, search_result.scores)
            move = search_result.moves[np.argmax(search_result.scores)]
            state.apply_move(move)
        else:
            result = bots[state.current_player()].step(state)
            state.apply_move(state.parent_game().get_move(result))
    return state


def _run_once(state: bridge.BridgeState2, bots: List[BlueChipBridgeBot], agent: SimpleAgent,
              pimc_bot: bridgelearn.PIMCBot, deal: Optional[Union[List[int], np.ndarray]] = None):
    for bot in bots:
        bot.restart()

    state_0 = state.clone()
    state_1 = state.clone()
    if deal is not None:
        assert_eq(len(deal), bridge.NUM_CARDS)
    else:
        deal = np.random.permutation(bridge.NUM_CARDS)
    for i in range(bridge.NUM_CARDS):
        move = game.get_chance_outcome(deal[i])
        state_0.apply_move(move)
        state_1.apply_move(move)

    state_0 = _bid_and_play(state_0, bots, agent, pimc_bot, [bridge.Seat.NORTH, bridge.Seat.SOUTH])
    state_1 = _bid_and_play(state_1, bots, agent, pimc_bot, [bridge.Seat.EAST, bridge.Seat.WEST])

    return state_0, state_1


if __name__ == '__main__':
    params = create_params(seed=23)
    game = create_bridge_game(params)
    bots = [
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.NORTH,
            controller_factory=controller_factory
        ),
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.EAST,
            controller_factory=controller_factory
        ),
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.SOUTH,
            controller_factory=controller_factory
        ),
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.WEST,
            controller_factory=controller_factory
        ),
    ]

    agent = SimpleAgent(DEFAULT_POLICY_CONF)
    agent.policy_net.load_state_dict(torch.load("sl/exp6/model8.pth")["state_dict"])
    agent.to("cuda")
    resampler = bridgelearn.UniformResampler(23)
    pimc_bot = bridgelearn.PIMCBot(resampler, 200)
    results = []
    for i_deal in range(num_deals):
        state_0, state_1 = _run_once(bridge.BridgeState2(game), bots, agent, pimc_bot)
        print("Deal #{}; final state:\n{}\n{}".format(i_deal, state_0, state_1))
        results.append(bridge.get_imp(state_0.scores()[0], state_1.scores()[0]))

    stats = np.array(results)
    print(stats)
    mean = np.mean(stats, axis=0)
    stderr = np.std(stats, axis=0, ddof=1) / np.sqrt(num_deals)
    print(u"Absolute score: {:+.1f}\u00b1{:.1f}".format(mean, stderr))
