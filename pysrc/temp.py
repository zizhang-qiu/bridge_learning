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

import re

def rank_str_to_rank(rank_str: str):
    if rank_str.isdigit():
        return int(rank_str) - 2
    if rank_str == "A":
        return 12
    if rank_str == "K":
        return 11
    if rank_str == "Q":
        return 10
    if rank_str == "J":
        return 9
    if rank_str == "T":
        return 8
    raise ValueError

def card_string_to_chance(card_str: str):
    suit_str = "CDHS"
    suit = suit_str.find(card_str[0])
    rank = rank_str_to_rank(card_str[1])
    return bridge.BridgeMove(bridge.MoveType.DEAL, bridge.Suit(suit), rank)

def call_string_to_move(call_str:str):
    if call_str.lower() == "pass":
        return bridge.BridgeMove(bridge.OtherCalls.PASS)
    if call_str.lower() == "double":
        return bridge.BridgeMove(bridge.OtherCalls.DOUBLE)
    if call_str.lower() == "redouble":
        return bridge.BridgeMove(bridge.OtherCalls.REDOUBLE)
    level = int(call_str[0])
    denomination = bridge.Denomination("CDHSN".find(call_str[1]))
    return bridge.BridgeMove(level, denomination)

def parse_state_string(s:str):
    player_str = "NESW"
    dealer_str = re.search("\'start\': \'(.*?)\'", s).group(1)
    dealer = player_str.find(dealer_str)
    print("dealer: ", dealer)
    game = bridge.BridgeGame({"dealer":str(dealer)})
    hands = re.findall("\'(.)\': \[(.*?)],", s)
    assert len(hands) == 4

    state = bridge.BridgeState(game)
    moves = [bridge.BridgeMove() for _ in range(bridge.NUM_CARDS)]
    for hand in hands:
        player = player_str.find(hand[0])
        print(player)
        card_strings = hand[1].split(",")
        card_strings = [card.strip()[1:-1] for card in card_strings]
        assert len(card_strings) == 13

        print(card_strings) 

        for i, card_str in enumerate(card_strings):
            index = player + i * bridge.NUM_PLAYERS
            move = card_string_to_chance(card_str)
            moves[index] = move

    for move in moves:
        state.apply_move(move)

    # print(state)

    bid_string = re.search("\'bid_list\': \[(.*?)]", s).group(1)

    bid_strings = bid_string.split(",")

    bid_strings = [bid.strip()[1:-1] for bid in bid_strings]
    print(bid_strings)
    
    for call in bid_strings:
        move = call_string_to_move(call)
        state.apply_move(move)
    
    print(state)
    
    encoder = bridge.CanonicalEncoder(game)
    for pl in range(4):
        obs = bridge.BridgeObservation(state, pl)
        f = encoder.encode(obs)
        print(f[:480])


# parse_state_string(
#     "{'start': 'W', 'W': ['C3', 'C8', 'C9', 'D5', 'DQ', 'H4', 'H5', 'H6', 'H8', 'HK', 'S2', 'SQ', 'SA'], 'N': ['C4', 'C6', 'C7', 'CT', 'CK', 'CA', 'D4', 'D6', 'DJ', 'H7', 'H9', 'HJ', 'HQ'], 'E': ['C5', 'CQ', 'D8', 'D9', 'HT', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'ST', 'SJ'], 'S': ['C2', 'CJ', 'D2', 'D3', 'D7', 'DT', 'DK', 'DA', 'H2', 'H3', 'HA', 'S3', 'SK'], 'bid_list': ['PASS', '1C', '3S', '3NT', 'DOUBLE', 'PASS', 'PASS', 'REDOUBLE', 'PASS', '5C', '5S', 'DOUBLE', 'PASS', 'PASS', 'PASS']}"
# )
parse_state_string(
    "{'start': 'W', 'W': ['C3', 'C4', 'CJ', 'DQ', 'DA', 'H2', 'H5', 'H6', 'H7', 'HQ', 'S7', 'SJ', 'SK'], 'N': ['C2', 'C5', 'CT', 'D3', 'D9', 'DJ', 'H3', 'H4', 'HT', 'HJ', 'S5', 'S9', 'SA'], 'E': ['C7', 'C8', 'CQ', 'CK', 'D2', 'D4', 'D6', 'DT', 'S2', 'S3', 'S4', 'S6', 'ST'], 'S': ['C6', 'C9', 'CA', 'D5', 'D7', 'D8', 'DK', 'H8', 'H9', 'HK', 'HA', 'S8', 'SQ'], 'bid_list': ['1H', 'PASS', '1S', 'DOUBLE', 'REDOUBLE', 'PASS', 'PASS', '2D', 'PASS', 'PASS', 'PASS']}"
)
