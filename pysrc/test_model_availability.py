"""
@author: qzz
@contact:q873264077@gmail.com
@file: test_model_availability.py
@time: 2024/07/28 20:03
"""

from typing import List
import torch
import time
from set_path import append_sys_path

append_sys_path()

import bridge
import pyrela
import bridgelearn

from agent import BridgeLSTMAgent
from create_bridge import create_bridge_game
from utils import load_dataset, load_rl_dataset
from common_utils import get_avg_and_sem

if __name__ == "__main__":
    test_set = load_rl_dataset("valid")
    print(test_set)

    num_threads = 1
    num_env_per_thread = 25
    # num_game_per_env = test_set["cards"].shape[0] // (num_threads * num_env_per_thread)
    num_game_per_env = 100
    print(num_game_per_env)

    bridge_dataset = bridgelearn.BridgeDataset(test_set["cards"], test_set["ddts"])  # type: ignore
    print(bridge_dataset.size())
    bridge_data = bridge_dataset.next()
    print(bridge_data.deal)

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

    agent_cons = lambda: BridgeLSTMAgent(
        device, in_dim, hid_dim, out_dim, num_priv_mlp_layer, num_publ_mlp_layer, num_lstm_layer, "gelu", 0.0, "publ-lstm",
        greedy
    )
    env_options = bridgelearn.BridgeEnvOptions()
    env_options.bidding_phase = True
    env_options.playing_phase = False

    env_actor_options = bridgelearn.EnvActorOptions()
    env_actor_options.eval = True

    # env = bridgelearn.DuplicateEnv({}, env_options, bridge_dataset)
    env = bridgelearn.BridgeEnv({}, env_options)
    env.set_bridge_dataset(bridge_dataset)
    env.reset()
    print(env)

    # sys.exit(0)
    #
    # print(env.feature(-1))
    # print(env.feature(1))
    # agent = agent_cons()

    # runner = rela.BatchRunner(agent, device, 1000, ["act", "get_h0"])
    # runner.start()

    # actor = bridgelearn.BridgePublicLSTMActor(runner, 0)

    # env.reset()

    # feature = env.feature()
    # print(feature)

    # print(env)
    # actor.reset(env)

    # actor.observe_before_act(env)

    # actor.act(env, 0)

    # print(env)

    st = time.perf_counter()
    context = pyrela.Context()

    runners = []
    for i in range(bridge.NUM_PLAYERS):
        agent = agent_cons()
        runner = pyrela.BatchRunner(agent, device, 1000, ["act", "get_h0"])
        runner.start()
        runners.append(runner)

    all_env_actors: List[bridgelearn.BridgeEnvActor] = []
    replay_buffer = pyrela.RNNPrioritizedReplay(int(1e5), 42, 0.0, 1.0, 2)
    for i_thread in range(num_threads):

        env_actors = []
        for i in range(num_env_per_thread):
            # env = bridgelearn.DuplicateEnv({}, env_options, bridge_dataset)
            env = bridgelearn.BridgeEnv({}, env_options)
            env.set_bridge_dataset(bridge_dataset)
            # actors = [
            #     bridgelearn.BridgePublicLSTMActor(runners[0])
            #     for i in range(bridge.NUM_PLAYERS)
            # ]
            actors = [
                bridgelearn.BridgePublicLSTMActor(
                    runners[0], 50, 1.0, replay_buffer, 0
                ),
                bridgelearn.BridgePublicLSTMActor(
                    runners[0], 50, 1.0, replay_buffer, 1
                ),
                bridgelearn.BridgePublicLSTMActor(
                    runners[0], 50, 1.0, replay_buffer, 2
                ),
                bridgelearn.BridgePublicLSTMActor(
                    runners[0], 50, 1.0, replay_buffer, 3
                ),
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

    print(replay_buffer.num_add())

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

    batch, weight = replay_buffer.sample(100, "cuda")
    batch_dict = batch.to_dict()
    for k, v in batch_dict.items():
        print(k, v)
