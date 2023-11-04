import argparse
import os
import pickle

import torch
from pprint import pformat
from torch.nn.functional import one_hot

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from net import MLP
from create_bridge import create_params
from set_path import append_sys_path
from common_utils.logger import Logger
from common_utils.saver import TopKSaver

append_sys_path()
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fc_layers", type=int, default=4)
    parser.add_argument("--hid_dim", type=int, default=2048)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_iterations", type=int, default=100000)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="sl/exp1")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    policy_net = MLP(num_fc_layers=args.num_fc_layers, hid_dim=args.hid_dim)
    policy_net.to(device=args.device)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    logger = Logger(os.path.join(args.save_dir, "log.txt"), auto_line_feed=True)

    saver = TopKSaver(args.save_dir, 10)
    logger.write(pformat(policy_net.get_conf()))
    opt = torch.optim.Adam(lr=args.lr, params=policy_net.parameters())
    params = create_params()
    game = bridge.BridgeGame(params)
    train_dataset = pickle.load(open(os.path.join(dataset_dir, "train.pkl"), "rb"))
    valid_dataset = pickle.load(open(os.path.join(dataset_dir, "valid.pkl"), "rb"))
    test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))

    train_generator = bridgelearn.SuperviseDataGenerator(train_dataset, args.batch_size, game, 42)
    valid_generator = bridgelearn.SuperviseDataGenerator(valid_dataset, 10, game, 0)
    valid_batch = valid_generator.all_data(args.device)

    for i in trange(1, args.num_iterations + 1):
        opt.zero_grad()
        batch = train_generator.next_batch(args.device)
        digits = policy_net(batch["s"])
        prob = torch.nn.functional.softmax(digits, -1)
        one_hot_label = one_hot(batch["label"] - bridge.NUM_CARDS, bridge.NUM_CALLS).to(args.device)
        # loss = -torch.mean(log_prob * one_hot_label)
        loss = torch.nn.functional.binary_cross_entropy(prob, one_hot_label.float())
        loss.backward()
        opt.step()

        # eval
        if i % args.eval_freq == 0:
            with torch.no_grad():
                digits = policy_net(valid_batch["s"])
                prob = torch.nn.functional.softmax(digits, -1)
                label = valid_batch["label"] - bridge.NUM_CARDS
                one_hot_label = one_hot(valid_batch["label"] - bridge.NUM_CARDS, bridge.NUM_CALLS).to(args.device)
                # loss = -torch.mean(log_prob * one_hot_label)
                loss = torch.nn.functional.binary_cross_entropy(prob, one_hot_label.float())
                acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
                print(f"Epoch {i // args.eval_freq}, acc={acc}, loss={loss}")

            saver.save(policy_net.get_save_dict(), -loss.item(), True)
