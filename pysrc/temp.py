import copy
import math
import multiprocessing
import os
import pickle
import random
from math import floor
from typing import Dict, List
import matplotlib.pyplot as plt
import timeit

import torch
import numpy as np
from tqdm import tqdm
from agent import BridgeAgent

from create_bridge import create_params
from common_utils.torch_utils import tensor_dict_to_device, optimizer_from_str
from common_utils import allocate_list_uniformly

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
from utils import extract_not_passed_out_trajectories, load_dataset, is_trajectory_not_passed_out

# dataset = load_dataset("D:/Projects/bridge_research/expert/train.txt")
# # print(len(dataset))

# bot = bridgeplay.TrajectoryBiddingBot(dataset, bridge.default_game)
# dataset = extract_not_passed_out_trajectories(dataset)
# # frequency = np.zeros(bridge.NUM_BIDS, dtype=np.int32)
# game = bridge.default_game

# def is_competitive(trajectory:List[int])->bool:
#     assert len(trajectory) > 56
#     first_bid = None
#     bid_partnership = None
#     for i, uid in enumerate(trajectory[:-52]):
#         if first_bid is None and bid_partnership is None and uid > 54:
#             first_bid = uid
#             bid_partnership = bridge.partnership(i - 52)

#         if first_bid is not None:
#             if uid == 53:
#                 return True

#             if uid > 54:
#                 if bridge.partnership(i-52) != bid_partnership:
#                     return True

#     return False


# def is_opener_declarer(trajectory: List[int]) -> bool:
#     first_bid = None
#     bid_partnership = None
#     for i, uid in enumerate(trajectory[:-52]):
#         if first_bid is None and bid_partnership is None and uid > 54:
#             first_bid = uid
#             bid_partnership = bridge.partnership(i - 52)

#     state = bridgeplay.construct_state_from_trajectory(trajectory, bridge.default_game)

#     declarer = state.get_contract().declarer

#     return bridge.partnership(declarer) == bid_partnership


# lengths = np.zeros(40, np.int32)
# # For each trajectory, check if the bot makes same bid.
# num_competitive = 0
# num_opener_is_declarer = 0
# for trajectory in tqdm(dataset):
#     # if is_competitive(trajectory):
#     #     num_competitive += 1
#     #     num_opener_is_declarer += int(is_opener_declarer(trajectory))
#     length = len(trajectory) - 104 if is_trajectory_not_passed_out(trajectory) else 4
#     lengths[length] += 1

#     # state = bridge.BridgeState(bridge.default_game)
#     # for uid in trajectory[:bridge.NUM_CARDS]:
#     #     chance = bridge.default_game.get_chance_outcome(uid)
#     #     state.apply_move(chance)
#     # while state.current_phase() == bridge.Phase.AUCTION:
#     #     move = bot.step(state)
#     #     state.apply_move(move)

#     # real_trajectory = trajectory[: -bridge.NUM_CARDS] if is_trajectory_not_passed_out(trajectory) else trajectory
#     # assert state.uid_history() == real_trajectory


# # for i, freq in enumerate(frequency):
# #     bid_str = repr(game.get_move(i + 3 + 52))
# #     print(f"{bid_str}: {freq}")

# # print(num_competitive)
# # print(num_opener_is_declarer)

# for i, l in enumerate(lengths):
#     print(i, l)

# print(np.sum(lengths[:21]) / np.sum(lengths))

# plt.bar(np.arange(40), lengths)
# plt.show()

print("ok" in "1ok2")

t1 = timeit.timeit("'ok' in '1ok2'")
t2 = timeit.timeit("'1ok2'.count('ok')")
print(t1)
print(t2)
