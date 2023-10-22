import argparse
import os
import pickle

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from create_bridge import create_params
from net import MLP
from common_utils import find_files_in_dir
from set_path import append_sys_path

append_sys_path()
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert")
    parser.add_argument("--model_dir", type=str, default="sl/exp1")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    params = create_params()
    game = bridge.BridgeGame(params)
    test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))
    test_generator = bridgelearn.SuperviseDataGenerator(test_dataset, 10, game, -1)
    test_batch = test_generator.all_data(args.device)
    model_paths = find_files_in_dir(args.model_dir, ".pth", 2)
    for p in model_paths:
        file = torch.load(p)
        policy_net = MLP.from_conf(file["conf"])
        policy_net.load_state_dict(file["state_dict"])
        policy_net.to(args.device)
        # test
        with torch.no_grad():
            digits = policy_net(test_batch["s"])
            prob = torch.nn.functional.softmax(digits, -1)
            label = test_batch["label"] - bridge.NUM_CARDS
            one_hot_label = one_hot(test_batch["label"] - bridge.NUM_CARDS, bridge.NUM_CALLS).to(args.device)
            loss = torch.nn.functional.binary_cross_entropy(prob, one_hot_label.float())
            acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
            print(f"{p}, acc={acc}, loss={loss}")
