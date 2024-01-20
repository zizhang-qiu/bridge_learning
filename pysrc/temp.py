import copy
import math
import multiprocessing
import pickle
import random
from math import floor
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from agent import BridgeAgent

from create_bridge import create_params
from common_utils.torch_utils import tensor_dict_to_device, optimizer_from_str

# print(torch.__version__)
# print(torch.cuda.is_available())
# import set_path
#
# set_path.append_sys_path()
from adan import Adan
import rela
import bridge
import bridgelearn

import bridgeplay
from agent import BridgeAgent, BridgeA2CModel
from net import MLP

# print(dir(bridgelearn))
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

# times = np.load(r"D:\Projects\bridge\evaluation\exp1\pimc_time.npy")
# print(times)
# print(np.mean(times))

# class StatManager:
#     def __init__(self):
#         self.stats = {}
#         self.locks = {}
#
#     def initialize_stat(self, key):
#         with multiprocessing.Lock():
#             self.stats[key] = multiprocessing.Value('i', 0)
#             self.locks[key] = multiprocessing.Lock()
#
#     def update_stat(self, key, value):
#         with self.locks[key]:
#             self.stats[key].value += value
#
#     def get_stats(self):
#         return {key: stat.value for key, stat in self.stats.items()}
#
#
# def worker(stat_manager, key, increment):
#     for _ in range(10):
#         stat_manager.update_stat(key, increment)




# model = BridgeA2CModel({}, {})
# model.to("cuda")
#


# import torch._C
#
# cmake_cxx_flags = []
# for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
#     val = getattr(torch._C, f"_PYBIND11_{name}")
#     # print(val, getattr(torch._C, f"_PYBIND11_{name}"))
#     if val is not None:
#         # print([f'-DPYBIND11_{name}=\\"{val}\\"'])
#         cmake_cxx_flags += [fr'-DPYBIND11_{name}=\"{val}\"']
# print(" ".join(cmake_cxx_flags), end="")

belief_net = MLP.from_conf(dict(
    activation_function="gelu",
    num_hidden_layers=4,
    input_size=480,
    output_size=3 * 52,
    hidden_size=1024
))

state_dict = torch.load("latest.pth")
# print(state_dict)
#
# print(belief_net.state_dict())
belief_net.load_state_dict(state_dict)

agent = BridgeA2CModel(
    policy_conf=dict(
        hidden_size=2048,
        num_hidden_layers=6,
        use_layer_norm=True,
        activation_function="gelu",
        use_dropout=True,
        dropout_prob=0.0
    ),
    value_conf=dict(
        hidden_size=2048,
        num_hidden_layers=6,
        use_layer_norm=True,
        activation_function="gelu",
        output_size=1
    ),
    belief_conf=dict(
        activation_function="gelu",
        num_hidden_layers=6,
        input_size=480,
        output_size=3 * 52,
        hidden_size=2048
    )
)
agent.policy_net.load_state_dict(torch.load("sl/exp6/model0.pthw"))
agent.belief_net.load_state_dict(torch.load("belief_sl/exp3/model2.pthw"))
agent.to("cuda")

batch_runner = rela.BatchRunner(agent, "cuda", 100, ["get_policy", "get_belief"])
batch_runner.start()
torch_actor = bridgeplay.TorchActor(batch_runner)

env = bridgelearn.BridgeEnv(params, False)

env.set_bridge_dataset(bridge_dataset)
env.reset_with_bridge_data()

while env.ble_state().current_phase() == bridge.Phase.AUCTION:
    f = env.feature()
    f["s"] = f["s"][:480]
    policy = torch_actor.get_policy(f)
    action_uid = policy["pi"].argmax() + 52
    env.step(action_uid)


print(env)

f = env.feature()
f["s"] = f["s"][:480]

policy = torch_actor.get_policy(f)
print(policy)

belief = torch_actor.get_belief(f)
print(belief)

# torch_sampler = bridgeplay.TorchActorResampler(torch_actor, bridge.default_game, 23)
#
# success_cnt = 0
# for i in range(1000):
#     res = torch_sampler.resample(env.ble_state())
#     if res.success:
#         success_cnt += 1
# print(success_cnt)
cfg = bridgeplay.TorchOpeningLeadBotConfig()
cfg.num_worlds = 20
cfg.num_max_sample = 1000
cfg.fill_with_uniform_sample = True
bot = bridgeplay.TorchOpeningLeadBot(torch_actor, bridge.default_game, 1, cfg)
move = bot.step(env.ble_state())
print(move)

dds_bot = bridgeplay.load_bot("dds", bridge.default_game, 0)
dds_move = dds_bot.step(env.ble_state())
print(dds_move)
moves = bridgeplay.dds_moves(env.ble_state())
print(moves)
# batcher = rela.Batcher(100)
# fut = batcher.send(f)
#
# print(fut.is_null())
#
# batch = batcher.get()
#
# batch = tensor_dict_to_device(batch, "cuda")
# print(batch)
# print(agent.policy_net)
# agent.policy_net(batch["s"])

# policy = agent.get_policy(tensor_dict_to_device(batch, "cuda"))

# print(policy)


#
# while env.ble_state().current_phase() == bridge.Phase.AUCTION:
#     pi = torch_actor.get_policy(env.feature())
#     action_uid = pi["pi"].argmax()
#     env.step(action_uid + 52)
# print(env)
# belief = torch_actor.get_belief(env.feature())
# print(belief)
# # print(bridge.CanonicalEncoder(env.ble_game()).encode(
# #     bridge.BridgeObservation(env.ble_state(), env.ble_state().current_player())))
# # resampler = bridgeplay.UniformResampler(1)
# torch.manual_seed(1)
# resampler = bridgeplay.TorchActorResampler(torch_actor, env.ble_game(), 1)
# for i in range(1000):
#     res = resampler.resample(env.ble_state())
#     # print(res)
#     # print(res.result)
#     print(res.success)
#     if res.success:
#         print(res.result)

# state2 = bridgeplay.construct_state_from_deal(res.result, env.ble_game())
# print(state2)

