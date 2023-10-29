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
bridge_dataset = bridgelearn.BridgeDataset(bridge.example_deals, bridge.example_ddts)
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
teams = [1, 2, 3, 4, 5, 6, 7, 8]


def allocate_once(items: List[int]):
    np.random.shuffle(items)
    result = []
    for i in range(len(items) // 2):
        pair = items[2 * i: 2 * i + 2]
        result.append(sorted(pair))
    return sorted(result, key=lambda x: x[0])


all_res = []
while True:
    res = allocate_once(items=teams)
    if res not in all_res:
        all_res.append(res)
    print(len(all_res))
