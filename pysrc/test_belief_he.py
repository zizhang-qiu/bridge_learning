"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: test_belief_he.py
@time: 2024/1/19 15:30
"""
import argparse
import os
import pickle
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_utils import Logger
from common_utils.file_utils import find_files_in_dir
from common_utils.torch_utils import activation_function_from_str
from compute_sl_metric import get_metrics
from create_bridge import create_params
from net import MLP
from pysrc.utils import extract_not_passed_out_trajectories
from set_path import append_sys_path
from train_belief import compute_hand_acc
from supervised_learn2 import BiddingDataset, cross_entropy, compute_accuracy

append_sys_path()
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="belief_sl/exp2")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    net_conf_path = os.path.join(args.file_dir, "net.yaml")
    with open(net_conf_path, "r") as f:
        net_conf = yaml.full_load(f)

    belief_net = MLP.from_conf(net_conf)
    belief_net.to(args.device)
    belief_net.eval()

    test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))
    test_dataset = extract_not_passed_out_trajectories(test_dataset)
    test_gen = bridgelearn.BeliefDataGen(test_dataset, args.batch_size, bridge.default_game)
    test_batch = test_gen.all_data(args.device)

    logger: Optional[Logger] = None
    if args.save:
        logger = Logger(os.path.join(args.file_dir, "test.txt"), True, auto_line_feed=True)

    state_dict_files = find_files_in_dir(args.file_dir, ".pth", 2)
    best_model = None
    best_acc = 0
    loss_func = torch.nn.MSELoss()
    for f in tqdm(state_dict_files):
        belief_net.load_state_dict(torch.load(f))
        with torch.no_grad():
            digits = belief_net(test_batch["s"])
            # prob = torch.nn.functional.sigmoid(digits)
            label = test_batch["belief_he"].to(args.device)
            print(digits[:10], label[:10], sep="\n")
            # loss = -torch.mean(log_prob * one_hot_label)
            loss = loss_func(digits, label)
            # acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
            # same_count, acc_per_player, acc = compute_hand_acc(prob, label)

            if logger is not None:
                logger.write(f"{f}, loss={loss.item()}")