import copy
import math
import multiprocessing
import os
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

valid_dataset = load_dataset("D:/Projects/bridge_research/expert/valid.txt")
print(len(valid_dataset))

bot = bridgeplay.TrajectoryBiddingBot(valid_dataset, bridge.default_game)

# For each trajectory, check if the bot makes same bid.
for trajectory in valid_dataset:
    state = bridge.BridgeState(bridge.default_game)
    for uid in trajectory[:bridge.NUM_CARDS]:
        chance = bridge.default_game.get_chance_outcome(uid)
        state.apply_move(chance)
    while state.current_phase() == bridge.Phase.AUCTION:
        move = bot.step(state)
        state.apply_move(move)

    real_trajectory = trajectory[: -bridge.NUM_CARDS] if is_trajectory_not_passed_out(trajectory) else trajectory
    assert state.uid_history() == real_trajectory
