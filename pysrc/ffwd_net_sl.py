import sys
import time

import numpy as np
import torch
from torch.cuda import device
from tqdm import tqdm, trange
from typing import List, Optional, Tuple, Callable
from argparse import ArgumentParser
import pprint
import os
from set_path import append_sys_path

append_sys_path()

import bridge
import pyrela
import bridgelearn

from common_utils import (
    Stopwatch,
    Logger,
    get_mem_usage,
    MultiStats,
    mkdir_with_increment,
    TopkSaver,
    find_files_in_dir
)
from utils import load_dataset, Tachometer, tensor_dict_to_device
from agent import BridgeLSTMAgent, BridgeFFWDAgent

BIDDING_ACTION_BASE = bridge.BIDDING_ACTION_BASE


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"D:\Projects\bridge_research\expert\train.txt",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="The eps for adam optimizer."
    )
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--encoder", type=str, default="detailed")
    parser.add_argument(
        "--capacity", type=int, default=int(800000), help="Capacity of replay buffer."
    )
    parser.add_argument("--prefetch", type=int, default=3)

    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--inf_loop", type=int, default=1)
    parser.add_argument("--reward_type", type=str, default="real", choices=["real", "dds"])
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--value_loss_weight", type=float, default=0.0)

    # Agent
    parser.add_argument("--p_in_dim", type=int, default=None)
    parser.add_argument("--v_in_dim", type=int, default=None)
    parser.add_argument("--p_hid_dim", type=int, default=1024)
    parser.add_argument("--v_hid_dim", type=int, default=1024)
    parser.add_argument("--p_out_dim", type=int, default=None)
    parser.add_argument("--num_p_mlp_layer", type=int, default=4)
    parser.add_argument("--num_v_mlp_layer", type=int, default=4)
    parser.add_argument("--p_activation", type=str, default="gelu")
    parser.add_argument("--v_activation", type=str, default="gelu")
    parser.add_argument("--net", type=str, choices=["ws", "sep"], default="sep")
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--save_dir", type=str, default="ffwd_sl")
    parser.add_argument("--eval_only", type=int, default=0,
                        help="Set to 1 if you only want to evaluate model.")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=r"D:\Projects\bridge_research\expert\test.txt",
    )

    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--load_model", type=str, default="")

    return parser.parse_args()


def compute_loss(pred_logits: torch.Tensor,
                 gt_action: torch.Tensor,
                 pred_values: torch.Tensor,
                 gt_reward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gt_action.dim() == 1
    labels = torch.nn.functional.one_hot(gt_action, bridge.NUM_CALLS)
    p_loss = -torch.sum(torch.log(pred_logits + 1e-10) * labels, -1)
    v_loss = torch.pow(pred_values - gt_reward, 2)
    return p_loss, v_loss


def compute_accuracy(pred_logits: torch.Tensor,
                     gt_action: torch.Tensor) -> torch.Tensor:
    assert gt_action.dim() == 1
    greedy_action = torch.argmax(pred_logits, 1)
    match = greedy_action == gt_action
    return match.float()


def evaluate(batches: List[pyrela.FFTransition],
             agent: BridgeFFWDAgent,
             args):
    # Eval.
    agent.eval()
    policy_loss_vec = []
    value_loss_vec = []
    acc_vec = []
    for batch in batches:
        with torch.no_grad():
            action = batch.action["a"].flatten().long().to(args.device)
            action[action >= BIDDING_ACTION_BASE] -= BIDDING_ACTION_BASE

            obs = tensor_dict_to_device(batch.obs, args.device)
            reward = batch.reward.to(args.device) / 7600
            reply = agent.forward(obs)

            pi = reply["pi"]
            v = reply["v"].squeeze() / 7600

            policy_loss, value_loss = compute_loss(pi, action, v, reward)
            acc = compute_accuracy(pi, action)
            policy_loss_vec.append(policy_loss)
            value_loss_vec.append(value_loss)
            acc_vec.append(acc)

    policy_loss = torch.cat(policy_loss_vec).mean()
    value_loss = torch.cat(value_loss_vec).mean()
    acc = torch.cat(acc_vec).mean()
    return policy_loss, value_loss, acc


def main():
    args = parse_args()
    eval_dataset = load_dataset(args.eval_dataset)
    print("Load eval dataset.")

    env_options = bridgelearn.BridgeEnvOptions()
    env_options.max_len = -1
    env_options.encoder = args.encoder

    env = bridgelearn.BridgeEnv({}, env_options)
    in_dim = env.feature_size()
    out_dim = env.max_num_action() - bridge.NUM_CARDS - 1  # No play actions and no-op action.
    print(f"in_dim：{in_dim}, out_dim:{out_dim}.")

    del env
    priority_exponent = 1.0
    priority_weight = 0.0
    replay_buffer = pyrela.FFPrioritizedReplay(args.capacity, args.seed, priority_exponent, priority_weight,
                                               args.prefetch)

    clone_data_generator = bridgelearn.FFCloneDataGenerator(replay_buffer,
                                                            args.num_threads,
                                                            env_options,
                                                            args.reward_type,
                                                            args.gamma)

    eval_batches = clone_data_generator.generate_eval_data(args.eval_batch_size, "cpu", eval_dataset)
    print("Eval batch created.")

    print("Creating agent...")
    agent = BridgeFFWDAgent(
        args.device,
        args.p_in_dim if args.p_in_dim is not None else in_dim,
        args.v_in_dim if args.v_in_dim is not None else in_dim,
        args.p_hid_dim,
        args.v_hid_dim,
        args.p_out_dim if args.p_out_dim is not None else out_dim,
        args.num_p_mlp_layer,
        args.num_v_mlp_layer,
        args.p_activation,
        args.v_activation,
        args.dropout,
        args.net
    ).to(args.device)
    opt = torch.optim.Adam(lr=args.lr, params=agent.parameters())

    # Eval mode.
    if bool(args.eval_only):
        assert args.load_model

        if os.path.isdir(args.load_model):
            model_paths = find_files_in_dir(args.load_model, "pthw")
        else:
            assert os.path.exists(args.load_model)
            model_paths = [args.load_model]

        agent.eval()
        for model_path in model_paths:
            ckpt = torch.load(model_path, map_location=args.device)
            agent.load_state_dict(ckpt["model_state_dict"])

            p_loss, v_loss, acc = evaluate(eval_batches, agent, args)
            print(f"Model: {model_path}, p_loss: {p_loss.item()}, v_loss: {v_loss.item()}, acc:{acc.item()}.")

        sys.exit(0)

    # Load state dict if provided.
    if args.load_model:
        assert os.path.exists(args.load_model)
        print(f"Continue training from ckpt {args.load_model}.")
        ckpt = torch.load(args.load_model, map_location=args.device)
        agent.load_state_dict(ckpt["model_state_dict"])
        if "opt_state_dict" in ckpt:
            opt.load_state_dict(ckpt["opt_state_dict"])

        # Eval.
        p_loss, v_loss, acc = evaluate(eval_batches, agent, args)
        print(f"Performance of ckpt model, p_loss: {p_loss.item()}, v_loss: {v_loss.item()}, acc:{acc.item()}.")

    train_dataset = load_dataset(args.dataset_path)
    print(f"Total num game: {len(train_dataset)}.")
    for i, game_trajectory in enumerate(train_dataset):
        clone_data_generator.add_game(game_trajectory)
        if (i + 1) % 10000 == 0:
            print(f"\r{i + 1} games added.", end="")
    print()

    stopwatch = Stopwatch()
    stats = MultiStats()
    global_stats = MultiStats()
    tachometer = Tachometer()

    args.save_dir = mkdir_with_increment(args.save_dir)
    logger = Logger(os.path.join(args.save_dir, "train.log"), verbose=True, auto_line_feed=True)
    saver = TopkSaver(args.save_dir, 5)
    logger.write(pprint.pformat(vars(args)))
    clone_data_generator.start_data_generation(bool(args.inf_loop), args.seed)

    while (n := replay_buffer.size()) < args.capacity:
        print(f"Warming up replay buffer, \r{n}/{args.capacity}", end="")
        time.sleep(1)
    print("\n", "Start training.", sep="")

    tachometer.start()
    # Train/eval loop.
    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch}, mem usage:\n{get_mem_usage()}")
        stopwatch.reset()
        stats.reset()
        agent.train()

        for i_batch in range(args.epoch_len):
            batch, weight = replay_buffer.sample(args.batch_size, args.device)
            torch.cuda.synchronize()
            stopwatch.time("Sample data")
            action = batch.action["a"].flatten().long()
            action[action >= BIDDING_ACTION_BASE] -= BIDDING_ACTION_BASE

            obs = batch.obs
            reply = agent.forward(obs)
            torch.cuda.synchronize()
            pi = reply["pi"]
            v = reply["v"].squeeze() / 7600
            reward = batch.reward / 7600

            policy_loss, value_loss = compute_loss(pi, action, v, reward)

            loss = (policy_loss + args.value_loss_weight * value_loss).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            torch.cuda.synchronize()
            stopwatch.time("Forward & backward")

            opt.step()
            opt.zero_grad()
            torch.cuda.synchronize()
            stopwatch.time("Update model.")

            replay_buffer.update_priority(weight.cpu())

            stats.feed("p_loss", policy_loss.mean().item())
            stats.feed("v_loss", value_loss.mean().item())
            stats.feed("loss", loss.item())

        for stat in stats.stats.values():
            print(stat.summary())

        stopwatch.summary()
        tachometer.lap(replay_buffer, args.batch_size * args.epoch_len, 1)
        # Eval.
        p_loss, v_loss, acc = evaluate(eval_batches, agent, args)
        logger.write(f"Epoch {epoch}, p_loss: {p_loss.item()}, v_loss: {v_loss.item()}, acc:{acc.item()}.")
        ckpt = {"model_state_dict": agent.state_dict(),
                "opt_state_dict": opt.state_dict}
        saver.save(None, ckpt, -p_loss.item())
        global_stats.feed("p_loss", p_loss.item())
        global_stats.feed("v_loss", v_loss.item())
        global_stats.feed("acc", acc.item())
        global_stats.save_all(args.save_dir, plot=True)
    clone_data_generator.terminate()


if __name__ == '__main__':
    main()
