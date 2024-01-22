"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: evaluate_opening_lead2.py
@time: 2024/1/22 9:36
"""
import argparse
import os
from typing import Tuple, Dict, OrderedDict, List

import torch
import yaml
import multiprocessing as mp

import common_utils
import set_path
from agent import BridgeA2CModel

set_path.append_sys_path()
import rela
import bridge
import bridgeplay
from train_belief import extract_available_trajectories


def load_net_conf_and_state_dict(model_dir: str, model_name: str, net_conf_filename: str = "net.yaml") \
        -> Tuple[Dict, OrderedDict]:
    with open(os.path.join(model_dir, net_conf_filename), "r") as f:
        conf = yaml.full_load(f)
    state_dict_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(state_dict_path)
    return conf, state_dict


def construct_deal_and_bidding_state(trajectory: List[int],
                                     game: bridge.BridgeGame = bridge.default_game) -> bridge.BridgeState:
    assert len(trajectory) > game.min_game_length()
    state = bridge.BridgeState(game)
    idx = 0
    while not state.current_phase() == bridge.Phase.PLAY:
        uid = trajectory[idx]
        if state.is_chance_node():
            move = game.get_chance_outcome(uid)
        else:
            move = game.get_move(uid)
        state.apply_move(move)
        idx += 1
    return state


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


class Worker(mp.Process):
    def __init__(self, flags: argparse.Namespace, trajectories: List[List[int]], q: mp.SimpleQueue,
                 process_idx: int = 0):
        super().__init__()
        self.args = flags
        self.trajectories = trajectories
        self.q = q
        self.process_idx = process_idx

    def run(self):
        dds_evaluator = bridgeplay.DDSEvaluator()

        # Create agent
        policy_conf, policy_state_dict = load_net_conf_and_state_dict(self.args.policy_model_dir,
                                                                      self.args.policy_model_name)
        belief_conf, belief_state_dict = load_net_conf_and_state_dict(self.args.belief_model_dir,
                                                                      self.args.belief_model_name)

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
        agent.to(self.args.device)
        print("Network loaded.")

        batch_runner = rela.BatchRunner(agent, self.args.device, 100, ["get_policy", "get_belief"])
        batch_runner.start()

        cfg = bridgeplay.TorchOpeningLeadBotConfig()
        cfg.num_worlds = self.args.num_worlds
        cfg.num_max_sample = self.args.num_max_sample
        cfg.fill_with_uniform_sample = bool(self.args.fill_with_uniform_sample)
        cfg.verbose = False

        torch_actor = bridgeplay.TorchActor(batch_runner)
        bot = bridgeplay.TorchOpeningLeadBot(torch_actor, bridge.default_game, 1, dds_evaluator, cfg)

        num_match = 0

        for j, trajectory in enumerate(self.trajectories):
            state = construct_deal_and_bidding_state(trajectory)
            assert not state.is_terminal()
            # Get dds moves.
            dds_moves = dds_evaluator.dds_moves(state)

            # Get bot's move
            bot_move = bot.step(state)

            if bot_move in dds_moves:
                num_match += 1
                self.q.put(1)
            else:
                self.q.put(0)

            print(f"Process {self.process_idx}, num match: {num_match}/{j + 1}, total: {len(self.trajectories)}")


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
    datasets = common_utils.allocate_list_uniformly(test_dataset, args.num_threads)

    queue = mp.SimpleQueue()

    workers = []
    for i in range(args.num_threads):
        worker = Worker(args, datasets[i], queue, i)
        workers.append(worker)

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    results = []
    while not queue.empty():
        item = queue.get()
        results.append(item)

    print(f"Final result: {sum(results)}/{len(results)}")
