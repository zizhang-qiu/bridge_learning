import copy
import pickle
from math import floor
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from agent import BridgeAgent

from create_bridge import create_params
from common_utils.torch_utils import tensor_dict_to_device

print(torch.__version__)
print(torch.cuda.is_available())
import set_path

set_path.append_sys_path()

import bridge
import bridgelearn
import rela
from agent import BridgeAgent, DEFAULT_VALUE_CONF, DEFAULT_POLICY_CONF

print(dir(bridgelearn))
params = create_params()
# deal = bridge.example_deals[0]
# ddt = bridge.example_ddts[0]
#
# env = bridgelearn.BridgeEnv(params, False)
# env.reset_with_deck_and_double_dummy_results(deal, ddt)
#
# bridge_dataset = bridgelearn.BridgeDataset(bridge.example_deals, bridge.example_ddts)


# print(bridge_dataset.size())
# data = bridge_dataset.next()
# print(data.deal, data.ddt)
#
# vec_env = bridgelearn.BridgeVecEnv()
# for i in range(100):
#     env = bridgelearn.BridgeEnv(params, False)
#     env.set_bridge_dataset(bridge_dataset)
#     vec_env.append(env)
# print(vec_env.size())
# vec_env.reset()
#
# vec_env.display(10)
# agent = BridgeAgent({}, {}).to("cuda")
#
# while not vec_env.all_terminated():
#     feature = vec_env.feature()
#     feature = tensor_dict_to_device(feature, "cuda")
#     print(feature)
#     reply = agent.act(feature)
#     reply = tensor_dict_to_device(reply, "cpu")
#     print(reply)
#     vec_env.step(reply)
#
# vec_env.display(10)
# feature = vec_env.feature()
# agent = BridgeAgent(DEFAULT_POLICY_CONF, DEFAULT_VALUE_CONF).to("cuda")
# print(type(agent))
# print(isinstance(agent, torch.jit.ScriptModule))
# runner = rela.BatchRunner(agent, "cuda", 10, ["act"])
# runner.set_log_freq(2)
# runner.start()
# game = bridge.BridgeGame(params)
#
# with open(r"D:\Projects\bridge_research\expert\train.pkl", "rb") as fp:
#     trajectories = pickle.load(fp)
# generator = bridgelearn.SuperviseDataGenerator(trajectories, 16, game, 1)
# for i in range(100):
#     batch = generator.next_batch("cuda")
#     print(batch)

# times = np.load(r"D:\Projects\bridge\evaluation\exp1\pimc_time.npy")
# print(times)
# print(np.mean(times))

trajectory = [32, 5, 33, 28, 7, 37, 46, 12, 45, 47, 25, 41, 15, 51, 31, 36, 6, 38, 43, 16, 19, 40, 24, 8, 11, 22, 48, 4,
              1, 26, 44, 14, 27, 17, 20, 18, 50, 30, 35, 49, 2, 23, 10, 0, 9, 21, 39, 13, 42, 34, 3, 29, 52, 52, 69, 52,
              52, 52, 16, 32, 40, 48, 3, 12, 19, 23, 38, 10, 18, 6, 5, 25, 29, 9, 36, 1, 21, 24, 49, 45, 17, 33, 8, 2,
              37, 20, 31, 28, 27, 51, 34, 46, 14, 50, 7, 47, 43, 4, 30, 35, 0, 42, 15, 26, 39, 41, 44, 13, 11, 22]
state = bridgelearn.construct_state_from_trajectory(trajectory, bridge.default_game)
print(state)