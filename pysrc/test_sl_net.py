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
import pprint
from typing import Optional

import numpy as np
import torch
import torchmetrics
import yaml
from pprint import pformat

from matplotlib import pyplot as plt
from torch.nn.functional import one_hot

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from net import MLP
from common_utils.torch_utils import activation_function_from_str, optimizer_from_str
from create_bridge import create_params
from common_utils import Logger
from supervised_learn2 import BiddingDataset, cross_entropy, compute_accuracy
from set_path import append_sys_path
from common_utils.file_utils import find_files_in_dir
from compute_sl_metric import get_metrics
from adan import Adan

append_sys_path()
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="sl/exp5")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert\sl_data")
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

    net_conf["activation_function"] = activation_function_from_str(net_conf["activation_function"])
    policy_net = MLP.from_conf(net_conf).to(args.device)
    policy_net.eval()

    params = create_params()
    game = bridge.BridgeGame(params)

    test_dataset = BiddingDataset(
        obs_path=os.path.join(args.dataset_dir, "test_obs.p"),
        label_path=os.path.join(args.dataset_dir, "test_label.p"),
    )
    print(len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logger:Optional[Logger] = None
    if args.save:
        logger = Logger(os.path.join(args.file_dir, "test.txt"), True, auto_line_feed=True)
    state_dict_files = find_files_in_dir(args.file_dir, ".pth", 2)
    best_model = None
    best_acc = 0
    for f in tqdm(state_dict_files):
        policy_net.load_state_dict(torch.load(f))
        with torch.no_grad():
            probs = []
            labels = []
            for s, label in test_loader:
                s = s.to(args.device)
                label = label.to(args.device)
                digits = policy_net(s)
                log_prob = torch.nn.functional.log_softmax(digits, -1)

                probs.append(torch.exp(log_prob))
                labels.append(label)

            probs = torch.cat(probs)
            labels = torch.cat(labels)
            acc = compute_accuracy(probs, labels)
            loss = cross_entropy(torch.log(probs), labels, bridge.NUM_CALLS)
            if acc > best_acc:
                best_acc = acc
                best_model = f
            if logger is not None:
                logger.write(f"{f}, acc={acc.item()}, loss={loss.item()}")

    policy_net.load_state_dict(torch.load(best_model))
    probs = []
    labels = []
    with torch.no_grad():
        for s, label in test_loader:
            s = s.to(args.device)
            label = label.to(args.device)
            digits = policy_net(s)
            log_prob = torch.nn.functional.log_softmax(digits, -1)
            probs.append(torch.exp(log_prob))
            labels.append(label)
    probs_tensor = torch.cat(probs)
    labels_tensor = torch.cat(labels)
    metrics = get_metrics(probs_tensor.cpu(), labels_tensor.cpu(), args.file_dir)
