"""
@author: qzz
@contact:q873264077@gmail.com
@file: temp3.py
@time: 2024/02/22 16:43
"""

import sys
import os
import re
import time
from typing import List
import hydra
import numpy as np
import matplotlib.pyplot as plt
import omegaconf
import torch
import pickle
from agent import BridgeA2CModel
import common_utils
from evaluate_declarer_play_against_wbridge5 import DuplicateSaveItem
import bridge
import bridgeplay
import rela
import bridgelearn
from utils import (
    load_net_conf_and_state_dict,
    tensor_dict_to_device,
    tensor_dict_unsqueeze,
    load_rl_dataset
)

# @hydra.main(config_path="conf/optimizer", config_name="adam", version_base="1.2")
# def main(args):
#         opt = hydra.utils.instantiate(args)
#         print(opt)

if __name__ == "__main__":
    # file_dir = "../declarer_eval/exp19"
    # execution_times = np.load(os.path.join(file_dir, "p1_execution_times.npy"))
    # print(common_utils.get_avg_and_sem(execution_times))
    # execution_times_arr = np.load(os.path.join(file_dir, "execution_times.npy"))
    # print(len(execution_times_arr))
    # print(common_utils.get_avg_and_sem(execution_times_arr))

    # with open(os.path.join(file_dir, "items"), "rb") as fp:

    #     duplicate_items: List[DuplicateSaveItem] = pickle.load(fp)

    # alpha_mu_tricks = [item.state0.num_declarer_tricks() for item in duplicate_items]
    # wb_tricks = [item.state1.num_declarer_tricks() for item in duplicate_items]
    # print(common_utils.get_avg_and_sem(alpha_mu_tricks))
    # print(common_utils.get_avg_and_sem(wb_tricks))

    # alpha_mu_scores = [
    #     item.state0.scores()[item.state0.get_contract().declarer]
    #     for item in duplicate_items
    # ]
    # wb_scores = [
    #     item.state1.scores()[item.state1.get_contract().declarer]
    #     for item in duplicate_items
    # ]
    # print(common_utils.get_avg_and_sem(alpha_mu_scores))
    # print(common_utils.get_avg_and_sem(wb_scores))

    # num_alpha_mu_win = 0
    # num_wb_win = 0
    # for item in duplicate_items:
    #     state0_win = item.state0.scores()[item.state0.get_contract().declarer] > 0
    #     state1_win = item.state1.scores()[item.state1.get_contract().declarer] > 0
    #     if state0_win != state1_win:
    #         if state0_win:
    #             num_alpha_mu_win += 1
    #         else:
    #             num_wb_win += 1

    # print(
    #     f"Num difference: {num_alpha_mu_win + num_wb_win}, num_alpha_mu_win: {num_alpha_mu_win}"
    # )

    #     #     main()
    #     tricks = [0, 0]
    #     scores = [0, 0]
    #     cnt = 0
    #     for i in range(4):
    #         with open(f"../declarer_eval/exp8/log_{i}.txt") as f:
    #             content = f.read()

    #         match = re.findall("tricks: (.*?) : (.*?)\n scores: (.*?):(.*?)\n", content)
    #         for m in match:
    #             tricks[0] += int(m[0])
    #             tricks[1] += int(m[1])
    #             scores[0] += int(m[2])
    #             scores[1] += int(m[3])
    #             cnt += 1

    #     print(tricks, scores, cnt)
    # arr = np.load("../declarer_eval/exp9/execution_times.npy")
    # print(arr)
    # print(common_utils.get_avg_and_sem(arr))
    # trajectory = "16 50 6 7 33 8 49 14 45 17 26 20 21 40 42 41 5 28 18 23 32 15 51 13 19 4 1 22 31 46 10 30 39 37 35 27 11 44 38 43 9 3 25 29 47 2 24 36 48 34 0 12 58 60 62 65 66 52 67 52 52 52 12 48 4 0 11 15 51 23 35 7 39 3 47 17 24 27 45 37 1 13 16 44 10 20 49 41 5 2 8 6 36 32 25 29 33 34 40 42 30 9 26 14 19 46 28 18 22 21 43 31 50 38"
    # trajectory = trajectory.split(" ")
    # trajectory = [int(a) for a in trajectory]

    # print(trajectory)

    # state = bridgeplay.construct_state_from_trajectory(trajectory, bridge.default_game)
    # print(state)

    # with open("../belief_sl/exp3/log.txt", "r") as f:
    #     content = f.read()
    # # print(content)

    # log_start = content.find("Epoch")

    # logs = content[log_start:].split("\n")

    # print(logs)

    # loss_list = []
    # acc_list = []
    # for log in logs[:20]:
    #     match = re.match("Epoch (.*?), loss=(.*?), acc=(.*?), model saved=.*?", log)
    #     if match:
    #         print(match.groups())
    #         epoch, loss, acc = match.groups()
    #         loss_list.append(float(loss))
    #         acc_list.append(float(acc))

    # fig, (ax1) = plt.subplots()
    # ax1: plt.Axes

    # lines1,  = ax1.plot(range(0, len(loss_list)), loss_list, label="loss")
    # ax1.set_ylabel("Cross Entropy Loss")
    # ax1.set_xlabel("Epoch")
    # ax1.set_xticks(range(0, len(loss_list), 3))

    # ax2: plt.Axes
    # ax2 = ax1.twinx()

    # lines2,  = ax2.plot(
    #     range(0, len(loss_list)), acc_list, linestyle="--", label="accuracy"
    # )
    # ax2.set_ylabel("Accuracy")
    # lines = [lines1, lines2]
    # labels = [line.get_label() for line in lines]
    # plt.legend(lines, labels, loc="right")

    # plt.show()

    from other_models import A2CAgent, PBEModel

    conf = omegaconf.OmegaConf.load("conf/jps_a2c/a2c.yaml")
    print(conf)

    belief_model_dir = "../belief_sl/exp3"
    belief_model_name = "model2.pthw"
    policy_model_dir = "../policy_sl/exp6"
    policy_model_name = "model0.pthw"
    device = "cuda"

    policy_conf, policy_state_dict = load_net_conf_and_state_dict(
        policy_model_dir, policy_model_name
    )
    belief_conf, belief_state_dict = load_net_conf_and_state_dict(
        belief_model_dir, belief_model_name
    )

    agent: A2CAgent = hydra.utils.instantiate(conf)
    agent = agent.to("cuda")
    agent2 = BridgeA2CModel(
        policy_conf=policy_conf,
        value_conf=dict(
            hidden_size=2048,
            num_hidden_layers=6,
            use_layer_norm=True,
            activation_function="gelu",
            output_size=1,
        ),
        belief_conf=belief_conf,
    )
    agent2.policy_net.load_state_dict(policy_state_dict)
    agent2.belief_net.load_state_dict(belief_state_dict)
    agent2.to(device)
    # agent2_clone = agent2.clone("cuda")
    # sys.exit(1)
    # print("Network loaded.")

    # runner = rela.BatchRunner(agent, device, 100, ["get_policy", "get_belief", "act"])
    # runner.start()

    # actor = bridgelearn.BridgeA2CActor(runner)

    options = bridgelearn.BridgeEnvOptions()
    options.bidding_phase = True
    options.playing_phase = False
    options.pbe_feature = True
    options.jps_feature = True
    options.dnns_feature = True

    # print(options.pbe_feature, options.jps_feature, options.dnns_feature)

    # dataset = bridgelearn.BridgeDataset(bridge.example_deals, bridge.example_ddts)
    # env = bridgelearn.DuplicateEnv({}, options, dataset)
    # env.reset()
    # print(env)
    # while not env.terminated():
    #     f = env.feature()

    # for i in range(74):
    #     env.reset()
    # print(env)

    # uids = [58, 52, 62, 52, 52, 53, 52, 52, 54, 52, 52]
    # for action in uids:
    #     env.step(action)
    # print(env)

    # feature = env.feature()
    # print(feature)

    # # print(env)

    # # print(feature)
    rl_dataset = load_rl_dataset("valid")
    cards = rl_dataset["cards"][:50000]
    ddts = rl_dataset["ddts"][:50000]
    dataset = bridgelearn.BridgeDataset(cards, ddts) # type: ignore
    env_actor_options = bridgelearn.EnvActorOptions()
    env_actor_options.eval = True
    actors: List[bridgelearn.Actor] = []
    runners = [
        rela.BatchRunner(
            agent2.clone("cuda") if i%2 == 0 else agent.clone("cuda"),
            "cuda",
            100000,
            ["act", "act_greedy"],
        )
        for i in range(4)
    ]
    num_threads = 1
    num_env_per_thread = 500
    envs: List[bridgelearn.BridgeEnvActor] = []
    num_game_per_env = len(cards) // (
        num_threads * num_env_per_thread
    )
    print(num_game_per_env)
    context = rela.Context()
    for i_thread in range(num_threads):

        for i_env in range(num_env_per_thread):
            actors = []
            env = bridgelearn.DuplicateEnv({}, options)
            env.set_bridge_dataset(dataset)
            env.reset()
            for i in range(4):
                actor = bridgelearn.BaselineActor(runners[i])
                # actor = bridgelearn.AllPassActor()
                actors.append(actor)
            env_actor = bridgelearn.BridgeEnvActor(env, env_actor_options, actors)
            envs.append(env_actor)
        t = bridgelearn.EnvActorThreadLoop(envs, num_game_per_env)  # type: ignore
        context.push_thread_loop(t)
    for runner in runners:
        runner.start()

    st = time.perf_counter()
    context.start()
    context.join()
    ed = time.perf_counter()
    print(ed - st)

    player1_rewards = []
    for e in envs:
        rewards = e.history_rewards()
        for r in rewards:
            player1_rewards.append(r[0])

    print(player1_rewards)
    print(len(player1_rewards))
    print(common_utils.get_avg_and_sem(player1_rewards))
    
    # for env in envs:
    #     print(env.history_info()[0])

    # for i, reward in enumerate(player1_rewards):

    #     if reward != 0:
    #         print(i)
    #         print(envs[0].history_info()[i])

    # for i in range(10):
    #     env_actor.observe_before_act()
    #     env_actor.act()
    #     env_actor.observe_after_act()
    #     env_actor.send_experience()
    #     env_actor.post_send_experience()
    #     print(env_actor.get_env())
    # print(env_actor.history_rewards())
    # print(env_actor)
    # actors[0].observe_before_act(env)
    # actors[0].act(env)
    # print(env)
    # f = env.feature()
    # print(f.keys())
    # f = env.feature()
    # print(f.keys())
    # f = env.feature()
    # print(f.keys())
    # reply = agent.act(
    #     tensor_dict_to_device(tensor_dict_unsqueeze(env.feature(), 0), "cuda")
    # )
    # print(reply)

    # future_reply : rela.FutureReply = runner.call("act", env.feature())
    # print(future_reply.is_null())
    # reply = future_reply.get()
    # print(reply)
    # reply = runners[0].block_call("act", tensor_dict_unsqueeze(env.feature(), 0))
    # print(reply)
