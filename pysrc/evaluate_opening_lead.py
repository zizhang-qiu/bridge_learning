"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: evaluate_opening_lead.py
@time: 2024/1/20 13:44
"""
import pickle
import pprint
import time
from typing import Dict, Tuple, OrderedDict, List

import yaml

import set_path
from agent import BridgeA2CModel

set_path.append_sys_path()
import torch

import argparse
from dataclasses import dataclass
import os
import multiprocessing as mp
import numpy as np
import logging
from loguru import logger
from train_belief import extract_available_trajectories
from net import MLP

import rela
import bridge
import bridgelearn
import bridgeplay
from common_utils.value_stats import MultiStats

GAME = bridge.default_game


def load_net_conf_and_state_dict(model_dir: str, model_name: str, net_conf_filename: str = "net.yaml") \
        -> Tuple[Dict, OrderedDict]:
    with open(os.path.join(model_dir, net_conf_filename), "r") as f:
        conf = yaml.full_load(f)
    state_dict_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(state_dict_path)
    return conf, state_dict


def evaluate_once(traj: List[int], bot: bridgeplay.PlayBot):
    state = bridgeplay.construct_state_from_deal(traj[:bridge.NUM_CARDS], GAME)
    action_idx = bridge.NUM_CARDS

    # Bidding.
    while state.current_phase() == bridge.Phase.AUCTION:
        move = GAME.get_move(traj[action_idx])
        state.apply_move(move)
        action_idx += 1

    # Opening lead.
    wbridge5_opening_lead = GAME.get_move(traj[action_idx])
    # print(wbridge5_opening_lead)
    bot_opening_lead = bot.step(state)
    # print(bot_opening_lead)
    dds_moves = bridgeplay.dds_moves(state)
    # print(dds_moves)

    return wbridge5_opening_lead in dds_moves, bot_opening_lead in dds_moves


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert")
    parser.add_argument("--policy_model_dir", type=str, default="sl/exp6")
    parser.add_argument("--policy_model_name", type=str, default="model0.pthw")
    parser.add_argument("--belief_model_dir", type=str, default="belief_sl/exp3")
    parser.add_argument("--belief_model_name", type=str, default="model2.pthw")

    parser.add_argument("--num_worlds", type=int, default=20)
    parser.add_argument("--num_max_sample", type=int, default=1000)
    parser.add_argument("--fill_with_uniform_sample", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load dataset
    with open(os.path.join(args.dataset_dir, "test.txt"), "r") as f:
        lines = f.readlines()
    test_dataset = []

    for i in range(len(lines)):
        line = lines[i].split(" ")
        test_dataset.append([int(x) for x in line])

    test_dataset = extract_available_trajectories(test_dataset)

    # Load networks
    policy_conf, policy_state_dict = load_net_conf_and_state_dict(args.policy_model_dir, args.policy_model_name)
    belief_conf, belief_state_dict = load_net_conf_and_state_dict(args.belief_model_dir, args.belief_model_name)

    agent = BridgeA2CModel(
        policy_conf=policy_conf,
        value_conf=dict(
            hidden_size=2048,
            num_hidden_layers=6,
            use_layer_norm=True,
            activation_function="gelu",
            output_size=1
        ),
        belief_conf=belief_conf
    )
    agent.policy_net.load_state_dict(policy_state_dict)
    agent.belief_net.load_state_dict(belief_state_dict)
    agent.to(args.device)
    print("Network loaded.")

    # Create torch actor
    batch_runner = rela.BatchRunner(agent, args.device, 100, ["get_policy", "get_belief"])
    batch_runner.start()
    torch_actor = bridgeplay.TorchActor(batch_runner)

    # Create bots.
    cfg = bridgeplay.TorchOpeningLeadBotConfig()
    cfg.num_worlds = args.num_worlds
    cfg.num_max_sample = args.num_max_sample
    cfg.fill_with_uniform_sample = bool(args.fill_with_uniform_sample)
    cfg.verbose = False
    torch_bot = bridgeplay.TorchOpeningLeadBot(torch_actor, bridge.default_game, 1, cfg)
    print("Bot created.")

    # Evaluate for each deal
    num_wbridge5_match = 0
    num_bot_match = 0
    for i, trajectory in enumerate(test_dataset):
        wbridge5_match, bot_match = evaluate_once(trajectory, torch_bot)
        num_wbridge5_match += int(wbridge5_match)
        num_bot_match += int(bot_match)
        print(f"wbridge5: {num_wbridge5_match}/{i + 1}, bot: {num_bot_match}/{i + 1}")

    print(f"num wbridge5 match: {num_wbridge5_match}/{len(test_dataset)}.")
    print(f"num bot match: {num_bot_match}/{len(test_dataset)}.")
