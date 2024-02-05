"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: temp2.py
@time: 2024/1/17 15:06
"""
import os
import pickle
import random

import torch
import torch.nn.functional as F
import yaml

from net import MLP
from train_belief import extract_available_trajectories

from set_path import append_sys_path

append_sys_path()

import bridge
import bridgelearn
import bridgeplay

# conf = yaml.full_load(open("belief_sl/exp2/net.yaml"))
#
# device = "cuda"
# batch_size = 10000
# net = MLP.from_conf(conf)
# net.load_state_dict(torch.load("belief_sl/exp2/model9.pthw"))
# net.to(device)
#
# dataset_dir = r"D:\Projects\bridge_research\expert"
#
# test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))
# test_dataset = extract_available_trajectories(test_dataset)[:10]
# test_gen = bridgelearn.BeliefDataGen(test_dataset, batch_size, bridge.default_game)
# test_batch = test_gen.all_data(device)
#
#
# digits = net(test_batch["s"])
#
# round_digits = torch.round(digits)
#
# print(digits)
#
# label = test_batch["belief_he"]
#
# for i in range(10):
#     print(round_digits[i], label[i], sep="\n")

# card = bridge.BridgeCard(bridge.Suit.CLUBS_SUIT, 2)
# print(card)
#
# with open("card", "wb") as fp:
#     pickle.dump(card, fp)
#
# with open("card", "rb") as fp:
#     card = pickle.load(fp)
#
# print(card)
#
# hand = bridge.BridgeHand()
# hand.add_card(card)
#
# print(hand)
# with open("hand", "wb") as fp:
#     pickle.dump(hand, fp)
#
# with open("hand", "rb") as fp:
#     hand = pickle.load(fp)
#
# print(hand)

game = bridge.BridgeGame({})


class TestBot(bridgeplay.PlayBot):
    def __init__(self):
        super().__init__()

    def step(self, state: bridge.BridgeState):
        return state.legal_moves()[0]


print(issubclass(TestBot, bridgeplay.PlayBot))

