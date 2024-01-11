"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: supervised_learn2.py
@time: 2024/1/10 17:43
"""
import argparse
import os
import pickle
from typing import Tuple

import numpy as np
import torch
import yaml
from pprint import pformat
from torch.nn.functional import one_hot

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from net import MLP
from common_utils.torch_utils import activation_function_from_str, optimizer_from_str, initialize_fc
from create_bridge import create_params
from common_utils import MultiStats
from set_path import append_sys_path
from common_utils.logger import Logger
from common_utils.saver import TopkSaver
from adan import Adan

append_sys_path()
import bridge
import bridgelearn


class BiddingDataset(Dataset):
    def __init__(self, obs_path: str, label_path: str):
        """
        The dataset contains bridge bidding data.
        Args:
            obs_path: The path of obs.
            label_path: The path of labels.
        """
        dataset = torch.load(obs_path)
        self.s = dataset["s"]
        # self.s: torch.Tensor = torch.hstack([dataset["s"], dataset["legal_actions"]])
        self.label: torch.Tensor = torch.load(label_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.s[index], self.label[index]


def cross_entropy(
        log_probs: torch.Tensor, label: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Compute cross entropy loss of given log probs and label.
    Args:
        log_probs: The log probs.
        label: The label, should be 1 dimensional.
        num_classes: The number of classes for one-hot.

    Returns:
        The cross entropy loss.
    """
    assert label.ndimension() == 1
    return -torch.mean(
        torch.nn.functional.one_hot(label.long(), num_classes) * log_probs
    )


def compute_accuracy(probs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy of given probs and label. Which is the number of highest value action equals with label
    divides number of all actions.
    Args:
        probs: The probs.
        label: The labels.

    Returns:
        The accuracy of prediction.
    """
    greedy_actions = torch.argmax(probs, 1)
    return (greedy_actions == label).int().sum() / greedy_actions.shape[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_conf", type=str, default="conf/net.yaml")
    parser.add_argument("--train_conf", type=str, default="conf/sl.yaml")
    parser.add_argument("--save_dir", type=str, default="sl/exp3")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert\sl_data")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    with open(args.train_conf, "r") as f:
        train_conf = yaml.full_load(f)
    with open(args.net_conf, "r") as f:
        net_conf = yaml.full_load(f)
    with open(os.path.join(args.save_dir, "net.yaml"), "w") as f:
        yaml.dump(net_conf, f)
    net_conf["activation_function"] = activation_function_from_str(net_conf["activation_function"])
    policy_net = MLP.from_conf(net_conf)
    initialize_fc(policy_net)
    policy_net.to(device=train_conf["device"])
    policy_net.train()

    logger = Logger(os.path.join(args.save_dir, "log.txt"), auto_line_feed=True)

    saver = TopkSaver(args.save_dir, 10)
    logger.write(pformat(net_conf))
    logger.write(pformat(train_conf))
    opt_cls = optimizer_from_str(train_conf["optimizer"], ["Adan"])
    opt = opt_cls(params=policy_net.parameters(), lr=train_conf["lr"], **train_conf["optimizer_args"])
    train_dataset = BiddingDataset(
        obs_path=os.path.join(args.dataset_dir, "train_obs.p"),
        label_path=os.path.join(args.dataset_dir, "train_label.p"),
    )
    valid_dataset = BiddingDataset(
        obs_path=os.path.join(args.dataset_dir, "valid_obs.p"),
        label_path=os.path.join(args.dataset_dir, "valid_label.p"),
    )

    train_loader = DataLoader(train_dataset, batch_size=train_conf["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_conf["valid_size"], shuffle=False)

    print(
        f"Load dataset successfully! Train dataset has {len(train_dataset)} samples. Valid dataset has {len(valid_dataset)} samples.")

    multi_stats = MultiStats()

    num_mini_batches = 0


    def evaluate() -> Tuple[float, float]:
        policy_net.eval()
        loss_list = []
        acc_list = []
        with torch.no_grad():
            for s, label in valid_loader:
                s = s.to(train_conf["device"])
                label = label.to(train_conf["device"])
                digits = policy_net(s)
                log_prob = torch.nn.functional.log_softmax(digits, -1)
                loss = cross_entropy(log_prob, label, bridge.NUM_CALLS)
                loss_list.append(loss.item())
                accuracy = compute_accuracy(torch.exp(log_prob), label)
                acc_list.append(accuracy.item())
        policy_net.train()
        return np.mean(loss_list).item(), np.mean(acc_list).item()


    while True:

        for s, label in train_loader:
            # print(s, label)
            num_mini_batches += 1
            opt.zero_grad()
            s = s.to(train_conf["device"])
            label = label.to(train_conf["device"])
            digits = policy_net(s)
            log_prob = torch.nn.functional.log_softmax(digits, -1)
            loss = cross_entropy(log_prob, label, bridge.NUM_CALLS)
            # loss = focal_loss_func(log_prob, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=args.grad_clip)
            opt.step()
            # eval
            if num_mini_batches % train_conf["eval_freq"] == 0:
                # multi_stats.save("train_loss", save_dir)
                eval_loss, acc = evaluate()
                multi_stats.feed("eval_loss", eval_loss)
                multi_stats.feed("accuracy", acc)
                multi_stats.save_all(args.save_dir, True)
                msg = f"checkpoint {(num_mini_batches + 1) // train_conf['eval_freq']}, eval loss={eval_loss}, accuracy={acc}."
                logger.write(msg)
                # save params
                saver.save(None, policy_net.state_dict(), acc, save_latest=True)
            if num_mini_batches == train_conf["num_iterations"]:
                break
