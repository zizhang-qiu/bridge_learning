"""
@author: qzz
@contact:q873264077@gmail.com
@file: test_model_availability.py
@time: 2024/07/28 20:03
"""
import random
import sys
from typing import List, Dict
import torch
import time

from set_path import append_sys_path

append_sys_path()

import bridge
import pyrela
import bridgelearn

from agent import BridgeLSTMAgent, BridgeFFWDAgent
from create_bridge import create_bridge_game
from utils import load_dataset, load_rl_dataset, tensor_dict_to_device
from common_utils import get_avg_and_sem, set_all_seeds


def analyze_vul(idx: int):
    side = idx // 2
    vul = (idx % 2)
    print(f"{['NS', 'EW'][side]}:{['vul', 'not vul'][vul]}.")


def analyze_opening_bid(idx: int, player_idx: int):
    idx -= 4
    real_player_idx = (idx + player_idx) % 4
    print(f"{'NESW'[real_player_idx]} makes a opening pass.")


def analyze_bidding_history(idx: int, player_idx: int):
    num_bits_per_bid = 4 * 6
    idx -= 8
    bid_index = idx // num_bits_per_bid
    bid_str = bridge.call_string(bid_index + 3)
    remainder = idx % num_bits_per_bid
    category = remainder // 4
    relative_player = remainder % 4
    action_str = ["make", "pass after make", "double", "pass after double",
                  "redouble", "pass after redouble"][category]
    real_player = (player_idx + relative_player) % 4
    print(f"Bid {bid_str}, {'NESW'[real_player]} {action_str}")


def analyze_hand(idx: int):
    idx -= 848
    suit = bridge.card_suit(idx)
    rank = bridge.card_rank(idx)
    print(f"{'CDHS'[suit]}{'23456789TJQKA'[rank]}, ", end="")


def analyze_detailed_encoding(feature: Dict[str, torch.Tensor],
                              player_idx: int,
                              no_vul=False):
    nonzero_indices = torch.nonzero(feature["s"], as_tuple=True)[0]
    for idx in nonzero_indices:
        # Vulnerability.
        if 0 <= idx < 4:
            if not no_vul:
                analyze_vul(idx)
        # Opening lead.
        elif 4 <= idx < 8:
            analyze_opening_bid(idx, player_idx)
        # Bidding history.
        elif 8 <= idx < 848:
            analyze_bidding_history(idx, player_idx)
        # Hand.
        elif idx < 900:
            analyze_hand(idx)
        elif idx == 900:
            print("This player's turn.")
    print()


if __name__ == "__main__":
    torch.set_printoptions(threshold=10000000000)
    set_all_seeds(12)
    test_set = load_rl_dataset("valid")
    # print(test_set)

    num_threads = 8
    num_env_per_thread = 10
    # num_game_per_env = test_set["cards"].shape[0] // (num_threads * num_env_per_thread)
    num_game_per_env = 40
    print(num_game_per_env)

    bridge_dataset = bridgelearn.BridgeDataset(test_set["cards"], test_set["ddts"])  # type: ignore
    # print(bridge_dataset.size())
    # bridge_data = bridge_dataset.next()
    # print(bridge_data.deal)

    # assert (
    #     num_env_per_thread * num_game_per_env * num_threads
    #     == test_set["cards"].shape[0]
    # )

    device = "cuda"
    in_dim = 900
    hid_dim = 1024
    out_dim = 39
    num_priv_mlp_layer = 4
    num_publ_mlp_layer = 2
    num_lstm_layer = 1
    greedy = True

    # agent_cons = lambda: BridgeLSTMAgent(
    #     device, in_dim, hid_dim, out_dim,
    #     num_priv_mlp_layer, num_publ_mlp_layer, num_lstm_layer,
    #     "gelu", 0.0, "lstm",
    #     greedy
    # )

    env_options = bridgelearn.BridgeEnvOptions()
    env_options.bidding_phase = True
    env_options.playing_phase = False
    # env_options.encoder = "detailed(turn=true)"
    env_options.encoder = "detailed"

    env_actor_options = bridgelearn.EnvActorOptions()
    env_actor_options.eval = True

    env = bridgelearn.DuplicateEnv({}, env_options, bridge_dataset)
    perf_size, priv_size, publ_size = env.feature_size()

    agent_cons = lambda: BridgeFFWDAgent(
        device, p_in_dim=priv_size, v_in_dim=perf_size,
        p_hid_dim=hid_dim, v_hid_dim=hid_dim,
        p_out_dim=out_dim - 1, num_p_mlp_layer=4,
        num_v_mlp_layer=4, p_activation="gelu",
        v_activation="gelu", dropout=0.0, net="sep",
        greedy=greedy, uniform_priority=True
    )
    # env = bridgelearn.BridgeEnv({}, env_options)
    # env.set_bridge_dataset(bridge_dataset)
    env.reset()

    # sys.exit(0)
    #
    # print(env.feature(-1))
    # print(env.feature(1))
    # agent = agent_cons()
    #
    # runner = pyrela.BatchRunner(agent, device, 1000, ["act", "get_h0"])
    # runner.start()
    # #
    # actor = bridgelearn.BridgeLSTMActor(runner, 0)

    # env.reset()

    # feature = env.feature(0)
    # print(feature)

    # while not env.terminated():
    #     print(env)
    #     legal_actions = env.legal_actions()
    #     print("legal_actions:", legal_actions)
    #     print("current_player: ", env.current_player())
    #     for p in range(4):
    #         f = env.feature(p)
    #         # print(f"player: {p}, feature:\n{f}")
    #         # print(f"nonzero: ", torch.nonzero(f["s"], as_tuple=True)[0])
    #         print("player ", p)
    #         print("legal_actions: ", torch.nonzero(f["legal_move"], as_tuple=True)[0])
    #         real_player = p if f["table_idx"].item() == 0 else (p + 1) % 4
    #         analyze_detailed_encoding(f, real_player, no_vul=True)
    #         if p == env.current_player():
    #             obs = {
    #                 "priv_s": f["s"],
    #                 "legal_move": f["legal_move"][-39:-1]
    #             }
    #             obs = tensor_dict_to_device(obs, device)
    #             action = agent.act(obs)
    #             print(action)
    #     action = random.choice(legal_actions)
    #     print("action: ", action, bridge.call_string(action - 52))
    #     env.step(action)
    #     # input("Click any button to continue.")
    #     # print(env)
    #
    # print(env)
    # print(env.rewards())

    # print(env)
    # actor.reset(env)
    #
    # actor.observe_before_act(env)
    #
    # actor.act(env, 0)
    #
    # print(env)

    # sys.exit(0)

    st = time.perf_counter()
    context = pyrela.Context()

    runners = []
    for i in range(bridge.NUM_PLAYERS):
        agent = agent_cons()
        agent.load_state_dict(torch.load("ffwd_sl/sl_sep.pthw", map_location="cuda")["model_state_dict"])
        agent.eval()
        runner = pyrela.BatchRunner(agent, device, 1000, ["act", "compute_priority"])
        runner.start()
        runners.append(runner)

    all_env_actors: List[bridgelearn.BridgeEnvActor] = []
    replay_buffer = pyrela.FFPrioritizedReplay(int(1e5), 42, 0.8, 0.6, 2)
    for i_thread in range(num_threads):

        env_actors = []
        for i in range(num_env_per_thread):
            env = bridgelearn.DuplicateEnv({}, env_options, bridge_dataset)
            # env = bridgelearn.BridgeEnv({}, env_options)
            # env.set_bridge_dataset(bridge_dataset)
            # actors = [
            #     bridgelearn.BridgePublicLSTMActor(runners[0])
            #     for i in range(bridge.NUM_PLAYERS)
            # ]
            actors = [
                bridgelearn.BridgeFFWDActor(
                    runners[0], 0
                ),
                # bridgelearn.BridgeFFWDActor(
                #     runners[0], 1
                # ),
                bridgelearn.AllPassActor(1),
                bridgelearn.BridgeFFWDActor(
                    runners[0], 2
                ),
                # bridgelearn.BridgeFFWDActor(
                #     runners[0], 3
                # ),
                bridgelearn.AllPassActor(3)
            ]
            env_actor = bridgelearn.BridgeEnvActor(env, env_actor_options, actors)  # type: ignore

            env_actors.append(env_actor)
        all_env_actors.extend(env_actors)

        thread_loop = bridgelearn.EnvActorThreadLoop(
            env_actors, num_game_per_env, i_thread, False
        )

        context.push_thread_loop(thread_loop)
    print(len(all_env_actors))

    context.start()
    context.join()

    ed = time.perf_counter()

    print(
        f"The simulation of {num_threads * num_env_per_thread * num_game_per_env} games takes {ed - st:.2f} seconds."
    )

    # print("num_add: ", replay_buffer.num_add())
    # print("size: ", replay_buffer.size())

    # Get stats.
    rewards = []
    infos = []
    for ea in all_env_actors:
        env_history_rewards = ea.history_rewards()
        env_history_info = ea.history_info()
        for r in env_history_rewards:
            rewards.append(r[0])
        infos.extend(env_history_info)
    for info in infos[:5]:
        print(info)

    print(f"rewards: {rewards}")
    print(f"mean & std: {get_avg_and_sem(rewards)}")

    # batch, weight = replay_buffer.sample(10, "cuda")
    # print("weight: ", weight)
    # batch_dict = batch.to_dict()
    # for k, v in batch_dict.items():
    #     print(k, v)
