"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: temp2.py
@time: 2024/1/17 15:06
"""
import os
import pickle

import torch
import torch.nn.functional as F
import yaml

from net import MLP
from train_belief import extract_available_trajectories

from set_path import append_sys_path
append_sys_path()
import bridgelearn
import bridge

conf = yaml.full_load(open("belief_sl/exp2/net.yaml"))

device = "cuda"
batch_size = 10000
net = MLP.from_conf(conf)
net.load_state_dict(torch.load("belief_sl/exp2/model9.pthw"))
net.to(device)

dataset_dir = r"D:\Projects\bridge_research\expert"

test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))
test_dataset = extract_available_trajectories(test_dataset)[:10]
test_gen = bridgelearn.BeliefDataGen(test_dataset, batch_size, bridge.default_game)
test_batch = test_gen.all_data(device)


digits = net(test_batch["s"])

round_digits = torch.round(digits)

print(digits)

label = test_batch["belief_he"]

for i in range(10):
    print(round_digits[i], label[i], sep="\n")


