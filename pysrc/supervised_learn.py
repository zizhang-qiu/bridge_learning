import argparse
import os
import pickle

import omegaconf
import torch
import yaml
from pprint import pformat
from torch.nn.functional import one_hot
import hydra
from tqdm import tqdm, trange

from common_utils.value_stats import MultiStats
from net import MLP
from common_utils.torch_utils import initialize_fc

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
    parser.add_argument("--save_dir", type=str, default="sl/exp3")
    parser.add_argument(
        "--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert"
    )
    return parser.parse_args()


@hydra.main("conf", "policy_sl", version_base="1.2")
def main(args):
   
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    policy_net = MLP.from_conf(args.net)
    initialize_fc(policy_net)
    policy_net.to(device=args.device)
    policy_net.train()

    logger = Logger(os.path.join(args.save_dir, "log.txt"), auto_line_feed=True)
    logger.write(omegaconf.OmegaConf.to_yaml(args))
    saver = TopkSaver(args.save_dir, args.topk)

    opt = hydra.utils.instantiate(args.optimizer, params=policy_net.parameters())
    params = create_params()
    game = bridge.BridgeGame(params)
    dataset_dir = args.dataset_dir
    train_dataset = pickle.load(open(os.path.join(dataset_dir, "train.pkl"), "rb"))
    valid_dataset = pickle.load(open(os.path.join(dataset_dir, "valid.pkl"), "rb"))
    test_dataset = pickle.load(open(os.path.join(dataset_dir, "test.pkl"), "rb"))

    train_generator = bridgelearn.SuperviseDataGenerator(
        train_dataset, args.batch_size, game, 42
    )
    valid_generator = bridgelearn.SuperviseDataGenerator(valid_dataset, args.valid_batch_size, game, 0)
    valid_batch = valid_generator.next_batch(args.device)

    stats = MultiStats()

    for i in trange(1, args["num_iterations"] + 1):
        torch.cuda.empty_cache()
        opt.zero_grad()
        batch = train_generator.next_batch(args["device"])
        digits = policy_net(batch["s"][:, :480])
        prob = torch.nn.functional.softmax(digits, -1)
        one_hot_label = one_hot(batch["label"] - bridge.NUM_CARDS, bridge.NUM_CALLS).to(
            args["device"]
        )
        # loss = -torch.mean(log_prob * one_hot_label)
        loss = torch.nn.functional.binary_cross_entropy(prob, one_hot_label.float())
        loss.backward()
        opt.step()

        # eval
        if i % args["eval_freq"] == 0:
            with torch.no_grad():
                policy_net.eval()
                digits = policy_net(valid_batch["s"][:, :480])
                prob = torch.nn.functional.softmax(digits, -1)
                label = valid_batch["label"] - bridge.NUM_CARDS
                one_hot_label = one_hot(label, bridge.NUM_CALLS).to(
                    args["device"]
                )
                # print(prob.shape)
                # print(one_hot_label.shape)
                # loss = -torch.mean(log_prob * one_hot_label)
                loss = torch.nn.functional.binary_cross_entropy(
                    prob, one_hot_label.float()
                )
                stats.feed("loss", loss.cpu().item())
                acc = (torch.argmax(prob, 1) == label).to(torch.float32).mean()
                stats.feed("acc", acc.cpu().item())

            saved = saver.save(
                policy_net, policy_net.state_dict(), -loss.item(), save_latest=True
            )
            logger.write(
                f"Epoch {i // args['eval_freq']}, acc={acc}, loss={loss}, model saved={saved}"
            )
            stats.save_all(args.save_dir, plot=True)
            policy_net.train()


if __name__ == "__main__":
    main()
