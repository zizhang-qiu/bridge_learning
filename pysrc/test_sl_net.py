"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: test_sl_net.py
@time: 2024/1/10 11:19
"""
import argparse
import os
import pickle

import torch
import yaml
from pprint import pformat
from torch.nn.functional import one_hot

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from net import MLP
from common_utils.torch_utils import activation_function_from_str, optimizer_from_str
from create_bridge import create_params
from set_path import append_sys_path
from common_utils.file_utils import find_files_in_dir
from adan import Adan

append_sys_path()
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="sl/exp3")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    net_conf_path = os.path.join(args.file_dir, "net.yaml")
    with open(net_conf_path, "r") as f:
        net_conf = yaml.full_load(f)

    net_conf["activation_function"] = activation_function_from_str(net_conf["activation_function"])
    policy_net = MLP.from_conf(net_conf).to(args.device)
    policy_net.eval()

    params = create_params()
    game = bridge.BridgeGame(params)
    test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))
    test_generator = bridgelearn.SuperviseDataGenerator(test_dataset, 10, game, 0)
    test_batch = test_generator.all_data(args.device)

    state_dict_files = find_files_in_dir(args.file_dir, ".pth", 2)
    for f in state_dict_files:
        policy_net.load_state_dict(torch.load(f))
        with torch.no_grad():
            digits = policy_net(test_batch["s"])
            prob = torch.nn.functional.softmax(digits, -1)
            label = test_batch["label"] - bridge.NUM_CARDS
            one_hot_label = one_hot(test_batch["label"] - bridge.NUM_CARDS, bridge.NUM_CALLS).to(
                args.device)
            # loss = -torch.mean(log_prob * one_hot_label)
            loss = torch.nn.functional.binary_cross_entropy(prob, one_hot_label.float())
            acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
            print(f"{f}, acc={acc}, loss={loss}")
