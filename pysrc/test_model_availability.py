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
import rela
import bridgelearn

from agent import BridgePublicLSTMagent
from create_bridge import create_bridge_game
from utils import load_dataset, load_rl_dataset
from common_utils import get_avg_and_sem

if __name__ == "__main__":
    test_set = load_rl_dataset("valid")
    # print(test_set)

    num_threads = 8
    num_env_per_thread = 25
    num_game_per_env = test_set["cards"].shape[0] // (num_threads * num_env_per_thread)
    # num_game_per_env = 100
    print(num_game_per_env)

    bridge_dataset = bridgelearn.BridgeDataset(test_set["cards"], test_set["ddts"])  # type: ignore

    # assert (
    #     num_env_per_thread * num_game_per_env * num_threads
    #     == test_set["cards"].shape[0]
    # )

    device = "cuda"
    in_dim = 480
    hid_dim = 1024
    out_dim = 38
    num_mlp_layer = 4
    num_lstm_layer = 2
    greedy = True

    agent_cons = lambda: BridgePublicLSTMagent(
        device, in_dim, hid_dim, out_dim, num_mlp_layer, num_lstm_layer, greedy
    )
    env_options = bridgelearn.BridgeEnvOptions()
    env_options.bidding_phase = True
    env_options.playing_phase = False

    env_actor_options = bridgelearn.EnvActorOptions()
    env_actor_options.eval = True

    # env = bridgelearn.DuplicateEnv({}, env_options, bridge_dataset)

    # agent = agent_cons()

    # runner = rela.BatchRunner(agent, device, 1000, ["act", "get_h0"])
    # runner.start()

    # actor = bridgelearn.BridgePublicLSTMActor(runner)

    # env.reset()

    # feature = env.feature()
    # print(feature)

    # print(env)
    # actor.reset(env)

    # actor.observe_before_act(env)

    # actor.act(env)

    # print(env)

    st = time.perf_counter()
    context = rela.Context()

    runners = []
    for i in range(bridge.NUM_PLAYERS):
        agent = agent_cons()
        runner = rela.BatchRunner(agent, device, 1000, ["act", "get_h0"])
        runner.start()
        runners.append(runner)
        
    all_env_actors:List[bridgelearn.BridgeEnvActor] = []
    for i_thread in range(num_threads):

        env_actors = []
        for i in range(num_env_per_thread):
            env = bridgelearn.DuplicateEnv({}, env_options, bridge_dataset)
            # actors = [
            #     bridgelearn.BridgePublicLSTMActor(runners[0])
            #     for i in range(bridge.NUM_PLAYERS)
            # ]
            actors = [
                bridgelearn.BridgePublicLSTMActor(runners[0], 0),
                bridgelearn.AllPassActor(1),
                bridgelearn.BridgePublicLSTMActor(runners[0], 2),
                bridgelearn.AllPassActor(3),
            ]
            env_actor = bridgelearn.BridgeEnvActor(env, env_actor_options, actors) # type: ignore

            env_actors.append(env_actor)
            all_env_actors.extend(env_actors)

        thread_loop = bridgelearn.EnvActorThreadLoop(
            env_actors, num_game_per_env, i_thread, True
        )

        context.push_thread_loop(thread_loop)

    context.start()
    context.join()

    ed = time.perf_counter()

    print(f"The simulation of {test_set['cards'].shape[0]} games takes {ed-st:.2f} seconds.")
    
    # Get stats.
    rewards = []
    for ea in all_env_actors:
        env_history_rewards = ea.history_rewards()
        for r in env_history_rewards:
            rewards.append(r[0])

    print(f"rewards: {rewards}")
    print(f"mean & std: {get_avg_and_sem(rewards)}")
        
