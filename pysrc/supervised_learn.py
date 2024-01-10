import argparse
import os
import pickle

import torch
import yaml
from pprint import pformat
from torch.nn.functional import one_hot

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from net import MLP
from common_utils.torch_utils import activation_function_from_str, optimizer_from_str, initialize_fc
from create_bridge import create_params
from set_path import append_sys_path
from common_utils.logger import Logger
from common_utils.saver import TopkSaver
from adan import Adan

append_sys_path()
import bridge
import bridgelearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_conf", type=str, default="conf/net.yaml")
    parser.add_argument("--train_conf", type=str, default="conf/sl.yaml")
    parser.add_argument("--save_dir", type=str, default="sl/exp2")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert")
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
    params = create_params()
    game = bridge.BridgeGame(params)
    train_dataset = pickle.load(open(os.path.join(dataset_dir, "train.pkl"), "rb"))
    valid_dataset = pickle.load(open(os.path.join(dataset_dir, "valid.pkl"), "rb"))
    test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))

    train_generator = bridgelearn.SuperviseDataGenerator(train_dataset, train_conf["batch_size"], game, 42)
    valid_generator = bridgelearn.SuperviseDataGenerator(valid_dataset, 10, game, 0)
    valid_batch = valid_generator.all_data(train_conf["device"])

    for i in trange(1, train_conf["num_iterations"] + 1):
        opt.zero_grad()
        batch = train_generator.next_batch(train_conf["device"])
        digits = policy_net(batch["s"])
        prob = torch.nn.functional.softmax(digits, -1)
        one_hot_label = one_hot(batch["label"] - bridge.NUM_CARDS, bridge.NUM_CALLS).to(train_conf["device"])
        # loss = -torch.mean(log_prob * one_hot_label)
        loss = torch.nn.functional.binary_cross_entropy(prob, one_hot_label.float())
        loss.backward()
        opt.step()

        # eval
        if i % train_conf["eval_freq"] == 0:
            with torch.no_grad():
                policy_net.eval()
                digits = policy_net(valid_batch["s"])
                prob = torch.nn.functional.softmax(digits, -1)
                label = valid_batch["label"] - bridge.NUM_CARDS
                one_hot_label = one_hot(valid_batch["label"] - bridge.NUM_CARDS, bridge.NUM_CALLS).to(
                    train_conf["device"])
                # loss = -torch.mean(log_prob * one_hot_label)
                loss = torch.nn.functional.binary_cross_entropy(prob, one_hot_label.float())
                acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()

            saved = saver.save(policy_net, policy_net.state_dict(), -loss.item(), save_latest=True)
            print(f"Epoch {i // train_conf['eval_freq']}, acc={acc}, loss={loss}, model saved={saved}")
            policy_net.train()
