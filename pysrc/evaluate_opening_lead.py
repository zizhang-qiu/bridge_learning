"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: evaluate_opening_lead.py
@time: 2024/1/20 13:44
"""
from typing import Dict, Tuple, OrderedDict

import yaml

import common_utils
import set_path
from agent import BridgeA2CModel

set_path.append_sys_path()
import torch

import argparse
import os
import numpy as np
from pysrc.utils import extract_not_passed_out_trajectories

import rela
import bridge
import bridgeplay

GAME = bridge.default_game


def load_net_conf_and_state_dict(model_dir: str, model_name: str, net_conf_filename: str = "net.yaml") \
        -> Tuple[Dict, OrderedDict]:
    with open(os.path.join(model_dir, net_conf_filename), "r") as f:
        conf = yaml.full_load(f)
    state_dict_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(state_dict_path)
    return conf, state_dict


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

    parser.add_argument("--num_threads", type=int, default=8)

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

    test_dataset = extract_not_passed_out_trajectories(test_dataset)
    datasets = common_utils.allocate_list_uniformly(test_dataset, args.num_threads)

    # Load networks
    # policy_conf, policy_state_dict = load_net_conf_and_state_dict(args.policy_model_dir, args.policy_model_name)
    # belief_conf, belief_state_dict = load_net_conf_and_state_dict(args.belief_model_dir, args.belief_model_name)
    #
    # agent = BridgeA2CModel(
    #     policy_conf=policy_conf,
    #     value_conf=dict(
    #         hidden_size=2048,
    #         num_hidden_layers=6,
    #         use_layer_norm=True,
    #         activation_function="gelu",
    #         output_size=1
    #     ),
    #     belief_conf=belief_conf
    # )
    # agent.policy_net.load_state_dict(policy_state_dict)
    # agent.belief_net.load_state_dict(belief_state_dict)
    # agent.to(args.device)
    # print("Network loaded.")

    dds_evaluator = bridgeplay.DDSEvaluator()
    # Create torch actor
    # batch_runner = rela.BatchRunner(agent, args.device, 100, ["get_policy", "get_belief"])
    # batch_runner.start()

    cfg = bridgeplay.BeliefBasedOpeningLeadBotConfig()
    cfg.num_worlds = args.num_worlds
    cfg.num_max_sample = args.num_max_sample
    cfg.fill_with_uniform_sample = bool(args.fill_with_uniform_sample)
    cfg.verbose = False
    q = bridgeplay.ThreadedQueueInt(int(1.25 * len(test_dataset)))
    context = rela.Context()
    for i in range(args.num_threads):
        # torch_actor = bridgeplay.TorchActor(batch_runner)
        # bot = bridgeplay.TorchOpeningLeadBot(torch_actor, bridge.default_game, 1, dds_evaluator, cfg)
        bot = bridgeplay.WBridge5TrajectoryBot(datasets[i], bridge.default_game)
        t = bridgeplay.OpeningLeadEvaluationThreadLoop(dds_evaluator, bot, bridge.default_game,
                                                       datasets[i], q, i, verbose=True)
        context.push_thread_loop(t)
    print("Threads created. Evaluation start.")

    context.start()
    context.join()

    res = []
    while not q.empty():
        num = q.pop()
        res.append(num)

    print(res)
    print(f"Num match: {np.sum(res)} / {len(res)}")
