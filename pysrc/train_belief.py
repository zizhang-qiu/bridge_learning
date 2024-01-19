"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: train_belief.py
@time: 2024/1/16 21:08
"""
import argparse
import os
import pickle
from pprint import pformat
from typing import Optional, List

import numpy as np
import torch
import yaml
from tqdm import trange

import set_path
from net import MLP
from common_utils import Logger, TopkSaver, optimizer_from_str
from create_bridge import create_params

set_path.append_sys_path()

import rela
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_conf", type=str, default="conf/belief_net.yaml")
    parser.add_argument("--train_conf", type=str, default="conf/sl.yaml")
    parser.add_argument("--save_dir", type=str, default="belief_sl/exp1")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert")
    return parser.parse_args()


def is_trajectory_available(trajectory: List[int]):
    return trajectory[-4:] != [bridge.OtherCalls.PASS.value + 52 for _ in range(bridge.NUM_PLAYERS)]


def extract_available_trajectories(trajectories: List[List[int]]) -> List[List[int]]:
    res = []
    for trajectory in trajectories:
        if is_trajectory_available(trajectory):
            res.append(trajectory)
    return res


def compute_hand_acc(pred: torch.Tensor, hand_label: torch.Tensor):
    assert pred.dim() == 2
    assert pred.shape == hand_label.shape
    same_count = np.zeros(shape=[pred.shape[0], bridge.NUM_PLAYERS - 1], dtype=np.int32)
    for relative_player in range(1, bridge.NUM_PLAYERS):
        current_pred = pred[:, (relative_player - 1) * bridge.NUM_CARDS:relative_player * bridge.NUM_CARDS].clone()
        current_label = label[:, (relative_player - 1) * bridge.NUM_CARDS:relative_player * bridge.NUM_CARDS].clone()
        assert current_label.shape[1] == bridge.NUM_CARDS
        assert current_pred.shape[1] == bridge.NUM_CARDS
        _, pred_cards = torch.topk(current_pred, bridge.NUM_CARDS_PER_HAND, dim=1)
        _, label_cards = torch.topk(current_label, bridge.NUM_CARDS_PER_HAND, dim=1)
        # print(pred_cards, label_cards, sep="\n")
        for j, (pred_cards_row, label_cards_row) in enumerate(zip(pred_cards, label_cards)):
            num_same = np.intersect1d(pred_cards_row.cpu().numpy(), label_cards_row.cpu().numpy()).shape[0]
            same_count[j, relative_player - 1] = num_same
    acc_per_player = np.mean(same_count / bridge.NUM_CARDS_PER_HAND, axis=0)
    overall_acc = np.mean(same_count / bridge.NUM_CARDS_PER_HAND)
    return same_count, acc_per_player, overall_acc


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

    belief_net = MLP.from_conf(net_conf)
    belief_net.to(device=train_conf["device"])
    belief_net.train()

    logger = Logger(os.path.join(args.save_dir, "log.txt"), auto_line_feed=True)
    logger.write(pformat(net_conf))
    logger.write(pformat(train_conf))

    saver = TopkSaver(args.save_dir, 10)

    params = create_params()
    game = bridge.BridgeGame(params)

    train_dataset = pickle.load(open(os.path.join(dataset_dir, "train.pkl"), "rb"))
    valid_dataset = pickle.load(open(os.path.join(dataset_dir, "valid.pkl"), "rb"))
    test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))

    # print(valid_dataset[0])
    train_dataset = extract_available_trajectories(train_dataset)
    valid_dataset = extract_available_trajectories(valid_dataset)

    print(
        f"Load dataset successfully, train set has {len(train_dataset)} samples, "
        f"valid set has {len(valid_dataset)} samples.")

    train_generator = bridgelearn.BeliefDataGen(train_dataset, train_conf["batch_size"], game)
    valid_generator = bridgelearn.BeliefDataGen(valid_dataset, train_conf["valid_batch_size"], game)
    valid_batch = valid_generator.all_data(train_conf["device"])

    opt_cls = optimizer_from_str(train_conf["optimizer"], ["Adan"])
    opt = opt_cls(params=belief_net.parameters(), lr=train_conf["lr"], **train_conf["optimizer_args"])
    loss_func = torch.nn.CrossEntropyLoss()
    # Main loop.

    for i in trange(1, train_conf["num_iterations"] + 1):
        torch.cuda.empty_cache()
        opt.zero_grad()
        batch = train_generator.next_batch(train_conf["device"])
        digits = belief_net(batch["s"])
        prob = torch.nn.functional.sigmoid(digits)
        label = batch["belief"].to(train_conf["device"])
        # loss = -torch.mean(log_prob * one_hot_label)
        loss = loss_func(prob, label)
        loss.backward()
        opt.step()

        # eval
        if i % train_conf["eval_freq"] == 0:
            with torch.no_grad():
                belief_net.eval()
                digits = belief_net(valid_batch["s"])
                prob = torch.nn.functional.sigmoid(digits)
                label = valid_batch["belief"].to(train_conf["device"])
                # loss = -torch.mean(log_prob * one_hot_label)
                loss = loss_func(prob, label)
                # acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
                same_count, acc_per_player, acc = compute_hand_acc(prob, label)
                # print(acc_per_player, acc, sep="\n")

            saved = saver.save(None, belief_net.state_dict(), acc.item(), save_latest=True)
            logger.write(f"Epoch {i // train_conf['eval_freq']}, loss={loss}, acc={acc.item()}, model saved={saved}")
            belief_net.train()
