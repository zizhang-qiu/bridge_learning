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
from typing import Optional
import sys
import numpy as np
import omegaconf
import torch
import yaml
from tqdm import trange
import hydra
import omegaconf
import common_utils
from utils import extract_not_passed_out_trajectories

import set_path
from net import MLP
from common_utils import Logger, TopkSaver, optimizer_from_str
from create_bridge import create_params
from agent import BridgeBeliefModel


set_path.append_sys_path()

import rela
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_conf", type=str, default="conf/belief_net.yaml")
    parser.add_argument("--train_conf", type=str, default="conf/sl.yaml")
    parser.add_argument("--save_dir", type=str, default="belief_sl/exp3")
    parser.add_argument(
        "--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert"
    )
    return parser.parse_args()


def compute_hand_acc(pred: torch.Tensor, hand_label: torch.Tensor):
    assert pred.dim() == 2
    assert pred.shape == hand_label.shape
    same_count = np.zeros(shape=[pred.shape[0], bridge.NUM_PLAYERS - 1], dtype=np.int32)
    for relative_player in range(1, bridge.NUM_PLAYERS):
        current_pred = pred[
            :,
            (relative_player - 1)
            * bridge.NUM_CARDS : relative_player
            * bridge.NUM_CARDS,
        ].clone()
        current_label = hand_label[
            :,
            (relative_player - 1)
            * bridge.NUM_CARDS : relative_player
            * bridge.NUM_CARDS,
        ].clone()
        assert current_label.shape[1] == bridge.NUM_CARDS
        assert current_pred.shape[1] == bridge.NUM_CARDS
        _, pred_cards = torch.topk(current_pred, bridge.NUM_CARDS_PER_HAND, dim=1)
        _, label_cards = torch.topk(current_label, bridge.NUM_CARDS_PER_HAND, dim=1)
        # print(pred_cards, label_cards, sep="\n")
        for j, (pred_cards_row, label_cards_row) in enumerate(
            zip(pred_cards, label_cards)
        ):
            num_same = np.intersect1d(
                pred_cards_row.cpu().numpy(), label_cards_row.cpu().numpy()
            ).shape[0]
            same_count[j, relative_player - 1] = num_same
    acc_per_player = np.mean(same_count / bridge.NUM_CARDS_PER_HAND, axis=0)
    overall_acc = np.mean(same_count / bridge.NUM_CARDS_PER_HAND)
    return same_count, acc_per_player, overall_acc


@hydra.main(config_path="conf", config_name="train_belief", version_base="1.2")
def main(args: omegaconf.DictConfig):
    # belief_net: MLP = hydra.utils.instantiate(args.belief_net)
    belief_net = MLP.from_conf(args.belief_net)
    belief_net.to(args.device)
    print(belief_net)
    common_utils.mkdir_if_not_exist(args.save_dir)
    logger = Logger(os.path.join(args.save_dir, "log.txt"), auto_line_feed=True)
    logger.write(omegaconf.OmegaConf.to_yaml(args))

    opt: torch.optim.Optimizer = hydra.utils.instantiate(
        args.optimizer, params=belief_net.parameters()
    )

    saver = TopkSaver(args.save_dir, args.top_k)

    params = create_params()
    game = bridge.BridgeGame(params)

    train_dataset = pickle.load(open(os.path.join(args.dataset_dir, "train.pkl"), "rb"))
    valid_dataset = pickle.load(open(os.path.join(args.dataset_dir, "valid.pkl"), "rb"))

    # print(valid_dataset[0])
    train_dataset = extract_not_passed_out_trajectories(train_dataset)
    valid_dataset = extract_not_passed_out_trajectories(valid_dataset)

    print(
        f"Load dataset successfully, train set has {len(train_dataset)} samples, "
        f"valid set has {len(valid_dataset)} samples.")

    # sys.exit(0)

    train_generator = bridgelearn.BeliefDataGen(train_dataset, args["batch_size"], game)
    valid_generator = bridgelearn.BeliefDataGen(valid_dataset, args["valid_batch_size"], game)
    valid_batch = valid_generator.all_data(args["device"])

    loss_func = torch.nn.CrossEntropyLoss()

    for i in trange(1, args["num_iterations"] + 1):
        torch.cuda.empty_cache()
        opt.zero_grad()
        batch = train_generator.next_batch(args["device"])
        digits = belief_net.forward(batch["s"])
        # loss = belief_net.loss(batch)
        # print(loss)
        digits = torch.nn.functional.sigmoid(digits)
        label = batch["belief"].to(args["device"])
        # # loss = -torch.mean(log_prob * one_hot_label)
        loss = loss_func(digits, label)
        loss.backward()
        opt.step()

        # eval
        if i % args["eval_freq"] == 0:
            with torch.no_grad():
                belief_net.eval()
                digits = belief_net(valid_batch["s"])
                digits = torch.nn.functional.sigmoid(digits)
                label = valid_batch["belief"].to(args["device"])
                # loss = -torch.mean(log_prob * one_hot_label)
                loss = loss_func(digits, label)
                # acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
                same_count, acc_per_player, acc = compute_hand_acc(digits, label)
                # print(acc_per_player, acc, sep="\n")

            saved = saver.save(None, belief_net.state_dict(), acc.item(), save_latest=True)
            logger.write(f"Epoch {i // args['eval_freq']}, loss={loss}, "
                         f"acc={acc.item()}, "
                         f"model saved={saved}")
            belief_net.train()


if __name__ == "__main__":
    main()
    # args = parse_args()
    # dataset_dir = args.dataset_dir
    # if not os.path.exists(args.save_dir):
    #     os.mkdir(args.save_dir)
    # with open(args.train_conf, "r") as f:
    #     train_conf = yaml.full_load(f)
    # with open(args.net_conf, "r") as f:
    #     net_conf = yaml.full_load(f)
    # with open(os.path.join(args.save_dir, "net.yaml"), "w") as f:
    #     yaml.dump(net_conf, f)

    # belief_net = MLP.from_conf(net_conf)

    # # net_conf["output_size"] = net_conf["hidden_size"]
    # # belief_net = BridgeBeliefModel(net_conf, 107)
    # belief_net.to(device=train_conf["device"])
    # belief_net.train()

    # logger = Logger(os.path.join(args.save_dir, "log.txt"), auto_line_feed=True)
    # logger.write(pformat(net_conf))
    # logger.write(pformat(train_conf))

    # saver = TopkSaver(args.save_dir, 10)

    # params = create_params()
    # game = bridge.BridgeGame(params)

    # train_dataset = pickle.load(open(os.path.join(dataset_dir, "train.pkl"), "rb"))
    # valid_dataset = pickle.load(open(os.path.join(dataset_dir, "valid.pkl"), "rb"))

    # # print(valid_dataset[0])
    # train_dataset = extract_not_passed_out_trajectories(train_dataset)
    # valid_dataset = extract_not_passed_out_trajectories(valid_dataset)

    # print(
    #     f"Load dataset successfully, train set has {len(train_dataset)} samples, "
    #     f"valid set has {len(valid_dataset)} samples.")

    # train_generator = bridgelearn.BeliefDataGen(train_dataset, train_conf["batch_size"], game)
    # valid_generator = bridgelearn.BeliefDataGen(valid_dataset, train_conf["valid_batch_size"], game)
    # valid_batch = valid_generator.all_data(train_conf["device"])

    # opt_cls = optimizer_from_str(train_conf["optimizer"], ["Adan"])
    # opt = opt_cls(params=belief_net.parameters(), lr=train_conf["lr"], **train_conf["optimizer_args"])
    # loss_func = torch.nn.CrossEntropyLoss()
    # # loss_func = torch.nn.MSELoss()
    # # Main loop.

    # for i in trange(1, train_conf["num_iterations"] + 1):
    #     torch.cuda.empty_cache()
    #     opt.zero_grad()
    #     batch = train_generator.next_batch(train_conf["device"])
    #     digits = belief_net.forward(batch["s"])
    #     # loss = belief_net.loss(batch)
    #     # print(loss)
    #     digits = torch.nn.functional.sigmoid(digits)
    #     label = batch["belief"].to(train_conf["device"])
    #     # # loss = -torch.mean(log_prob * one_hot_label)
    #     loss = loss_func(digits, label)
    #     loss.backward()
    #     opt.step()

    #     # eval
    #     if i % train_conf["eval_freq"] == 0:
    #         with torch.no_grad():
    #             belief_net.eval()
    #             digits = belief_net(valid_batch["s"])
    #             digits = torch.nn.functional.sigmoid(digits)
    #             label = valid_batch["belief"].to(train_conf["device"])
    #             # loss = -torch.mean(log_prob * one_hot_label)
    #             loss = loss_func(digits, label)
    #             # acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
    #             same_count, acc_per_player, acc = compute_hand_acc(digits, label)
    #             # print(acc_per_player, acc, sep="\n")

    #         saved = saver.save(None, belief_net.state_dict(), acc.item(), save_latest=True)
    #         logger.write(f"Epoch {i // train_conf['eval_freq']}, loss={loss}, "
    #                      f"acc={acc.item()}, "
    #                      f"model saved={saved}")
    #         belief_net.train()
