import sys
import time

import numpy as np
import torch
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
from utils import load_dataset, Tachometer
from agent import BridgeLSTMAgent

BIDDING_ACTION_BASE = bridge.BIDDING_ACTION_BASE


def get_bidding_length(data: List[int]) -> int:
    if len(data) == bridge.NUM_CARDS + bridge.NUM_PLAYERS:
        return bridge.NUM_PLAYERS
    else:
        return len(data) - 2 * bridge.NUM_CARDS  # Deal and play.


def get_max_len(*datasets: List[List[int]], len_func: Callable[[List[int]], int] = len):
    res = 0
    for dataset in datasets:
        lengths = [len_func(data) for data in dataset]
        res = max(res, np.max(lengths))

    return res


def create_data_generator(
        trajectories: List[List[int]],
        max_len: int,
        num_threads: int,
        replay_buffer_capacity: int,
        prefetch: int,
        seed: int,
) -> Tuple[bridgelearn.CloneDataGenerator, pyrela.RNNPrioritizedReplay]:
    """Create data generator and replay buffer.

    Args:
        trajectories (List[List[int]]): The list of game trajectories.
        max_len (int): The max_len for RNNTransition.
        num_threads (int): Number of threads used for generating data.
        replay_buffer_capacity (int): The capacity of replay buffer.
        prefetch (int): The prefetch of replay buffer.
        seed (int): Random seed.

    Returns:
        Tuple[bridgelearn.CloneDataGenerator, pyrela.RNNPrioritizedReplay]: Clone data generator and replay buffer.
    """
    print(f"total num game: {len(trajectories)}")
    # priority not used
    priority_exponent = 1.0
    priority_weight = 0.0
    replay_buffer = pyrela.RNNPrioritizedReplay(
        replay_buffer_capacity, seed, priority_exponent, priority_weight, prefetch
    )

    data_gen = bridgelearn.CloneDataGenerator(replay_buffer, max_len, num_threads)

    game_params = {}

    data_gen.set_game_params(game_params)

    for i, game_trajectory in enumerate(trajectories):
        data_gen.add_game(game_trajectory)
        if (i + 1) % 10000 == 0:
            print(f"\r{i + 1} games added.", end="")

    return data_gen, replay_buffer


def compute_loss(
        pred_logits: torch.Tensor,
        legal_move: torch.Tensor,
        ground_truth_action: torch.Tensor,
        mask: torch.Tensor,
        no_op_actions_weight: float = 1.0,
):
    seq_len, bsize, num_actions = pred_logits.size()
    assert (
            pred_logits.size() == legal_move.size()
    ), f"size not match, {pred_logits.size()} vs {legal_move.size()}"
    # pred_logits = pred_logits * legal_move
    pred_logits = pred_logits - (1 - legal_move) * 1e10
    pred_logits = pred_logits.reshape(-1, num_actions)

    ground_truth_action = ground_truth_action.flatten()
    loss = torch.nn.functional.cross_entropy(
        pred_logits, ground_truth_action, reduction="none"
    )
    loss = loss.view(seq_len, bsize)
    mask2 = mask.clone()
    mask2[(ground_truth_action == num_actions - 1).view(seq_len, bsize)] = (
        no_op_actions_weight
    )

    loss = (loss * mask2).sum(0).mean()
    return loss


def compute_accuracy(
        pred_logits: torch.Tensor,
        legal_move: torch.Tensor,
        ground_truth_action: torch.Tensor,
        mask: torch.Tensor,
):
    seq_len, bsize, num_actions = pred_logits.size()
    assert (
            pred_logits.size() == legal_move.size()
    ), f"size not match, {pred_logits.size()} vs {legal_move.size()}"
    # pred_logits = pred_logits * legal_move
    pred_logits = pred_logits - (1 - legal_move) * 1e10
    pred_logits = pred_logits.reshape(-1, num_actions)
    ground_truth_action = ground_truth_action.flatten()
    # print("pred_action: ", pred_logits.argmax(dim=1))
    # print("gt action: ", ground_truth_action)
    accuracy = (pred_logits.argmax(dim=1) == ground_truth_action).float()
    # print(accuracy)
    # print(accuracy.size())
    accuracy = accuracy.view(seq_len, bsize)
    accuracy1 = (accuracy * mask).sum() / (mask.sum())
    # print(mask)
    mask2 = mask.clone()
    mask2[(ground_truth_action == num_actions - 1).view(seq_len, bsize)] = 0
    # print(mask)
    accuracy_without_no_op = (accuracy * mask2).sum() / (mask2.sum())
    return accuracy1, mask.sum(), accuracy_without_no_op, mask2.sum()


def train(
        model: BridgeLSTMAgent,
        device: str,
        opt: torch.optim.Optimizer,
        replay_buffer: pyrela.RNNPrioritizedReplay,
        batch_size: int,
        num_batch: int,
        grad_clip: float,
        no_op_actions_weight: float,
        stopwatch: Optional[Stopwatch],
        stats: MultiStats,
):
    for i_batch in trange(num_batch):
        batch, weight = replay_buffer.sample(batch_size, device)
        # print(batch.seq_len)

        priv_s = batch.obs["priv_s"]
        publ_s = batch.obs["publ_s"]
        legal_move = batch.obs["legal_move"]
        # [seq_len, batch]
        action = batch.action["a"]
        action[action >= BIDDING_ACTION_BASE] -= BIDDING_ACTION_BASE

        mask = torch.arange(0, priv_s.size(0), device=batch.seq_len.device)
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        if stopwatch is not None:
            torch.cuda.synchronize()
            stopwatch.time("sample data")

        reply = model.forward(
            dict(priv_s=priv_s, publ_s=publ_s, legal_move=legal_move, action=action)
        )
        pi = reply["pi"]
        legal_move = legal_move[:, :, -model.out_dim:]

        loss = compute_loss(pi, legal_move, action, mask, no_op_actions_weight)
        # print(f"loss: {loss.item()}")
        loss.backward()
        if stopwatch is not None:
            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

        g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad()

        replay_buffer.update_priority(weight.cpu())

        if stopwatch is not None:
            stopwatch.time("update model")

        accuracy, num_eval_actions, accuracy_without_no_op, num_eval_actions_without_no_op = compute_accuracy(
            pi, legal_move, action, mask
        )

        stats.feed("loss", loss.item())
        stats.feed("grad_norm", g_norm.item())
        stats.feed("accuracy", accuracy.item())
        stats.feed("accuracy_without_no_op", accuracy_without_no_op.item())
    return


def evaluate(
        agent: BridgeLSTMAgent,
        eval_batch: List[pyrela.RNNTransition],
        args
):
    stats_ = MultiStats()
    for batch in tqdm(eval_batch):
        # batch.to_device(args.device)
        priv_s = batch.obs["priv_s"].to(args.device)
        publ_s = batch.obs["publ_s"].to(args.device)
        legal_move = batch.obs["legal_move"].to(args.device)
        # [seq_len, batch]
        action = batch.action["a"].to(args.device)
        action[action >= BIDDING_ACTION_BASE] -= BIDDING_ACTION_BASE

        seq_len = batch.seq_len.to(args.device)

        mask = torch.arange(0, priv_s.size(0), device=action.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        reply = agent.forward(
            dict(priv_s=priv_s, publ_s=publ_s, legal_move=legal_move, action=action)
        )
        pi = reply["pi"]
        legal_move = legal_move[:, :, -agent.out_dim:]
        loss = compute_loss(pi, legal_move, action, mask, args.no_op_actions_weight)
        accuracy, num_eval_actions, accuracy_without_no_op, num_eval_actions_without_no_op = compute_accuracy(
            pi, legal_move, action, mask
        )
        # batch.to_device("cpu")
        torch.cuda.empty_cache()
        stats_.feed("loss", loss.item())
        stats_.feed("accuracy", accuracy.item())
        stats_.feed("accuracy_without_no_op", accuracy_without_no_op.item())
        stats_.feed("num_eval_actions", num_eval_actions.item())
        stats_.feed("num_eval_actions_without_no_op", num_eval_actions_without_no_op.item())

    # print(f"Done. Eval result for model {args.load_model}:")
    # for name, stat in stats.stats.items():
    #     print(stat.summary())
    return stats_


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"D:\Projects\bridge_research\expert\train.txt",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="The eps for adam optimizer."
    )
    parser.add_argument("--num_threads", type=int, default=6)
    parser.add_argument(
        "--capacity", type=int, default=int(2e4), help="Capacity of replay buffer."
    )
    parser.add_argument("--prefetch", type=int, default=3)

    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--inf_loop", type=int, default=1)

    parser.add_argument("--net", type=str, choices=["publ-lstm", "lstm"], default="lstm")
    parser.add_argument("--hid_dim", type=int, default=1024)
    parser.add_argument("--num_priv_mlp_layer", type=int, default=4)
    parser.add_argument("--num_publ_mlp_layer", type=int, default=2)
    parser.add_argument("--num_lstm_layer", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--no_op_actions_weight",
        type=float,
        default=1.0,
        help="The weight to compute loss on no op actions.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--save_dir", type=str, default="lstm_sl")
    parser.add_argument("--eval_only", type=int, default=1)
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=r"D:\Projects\bridge_research\expert\test.txt",
    )

    parser.add_argument("--eval_batch_size", type=int, default=400)
    parser.add_argument("--load_model", type=str, default="lstm_sl/exp5")

    return parser.parse_args()


if __name__ == "__main__":
    # os.environ['OMP_NUM_THREADS'] = "1"
    torch.set_printoptions(threshold=100000)
    args = parse_args()

    eval_dataset = load_dataset(args.eval_dataset)
    print("Load eval dataset.")

    assert bridge.NUM_PLAYERS * len(eval_dataset) % args.eval_batch_size == 0

    env = bridgelearn.BridgeEnv({}, bridgelearn.BridgeEnvOptions())
    in_dim = env.feature_size()
    out_dim = env.max_num_action() - bridge.NUM_CARDS  # No play actions.
    print(f"in_dim：{in_dim}, out_dim:{out_dim}")

    del env

    if bool(args.eval_only):
        max_len = get_max_len(eval_dataset, len_func=get_bidding_length)
        clone_data_generator = bridgelearn.CloneDataGenerator(None, max_len, 1)  # type: ignore
        eval_batch = clone_data_generator.generate_eval_data(
            args.eval_batch_size,
            "cpu",  # Put batches in cpu here due to gpu usage.
            eval_dataset,
        )

        del clone_data_generator
        print("Start eval.")
        assert args.load_model
        if os.path.isdir(args.load_model):
            model_paths = find_files_in_dir(args.load_model, "pthw")
        elif os.path.isfile(args.load_model):
            model_paths = [args.load_model]
        else:
            raise ValueError(f"load_model {args.load_model} is neither a directory nor a file.")
        for model_path in model_paths:
            print(f"Model {model_path:}")
            checkpoint = torch.load(model_path)
            agent = BridgeLSTMAgent(
                args.device,
                in_dim,
                args.hid_dim,
                out_dim,
                args.num_priv_mlp_layer,
                args.num_publ_mlp_layer,
                args.num_lstm_layer,
                args.activation,
                args.dropout,
                args.net
            )
            agent.load_state_dict(checkpoint["model_state_dict"])
            agent.eval()
            # print("Load trained model.")
            with torch.no_grad():
                eval_stats = evaluate(agent, eval_batch, args)

            # print(f"Done. Eval result for model {args.load_model}:")
            for name, stat in eval_stats.stats.items():
                print(stat.summary())

            accuracy_without_no_op = (
                    (np.array(eval_stats.get("accuracy_without_no_op").stat_list)
                     * np.array(eval_stats.get("num_eval_actions_without_no_op").stat_list)).sum(0)
                    / np.array(eval_stats.get("num_eval_actions_without_no_op").stat_list).sum(0))

            accuracy = (
                    (np.array(eval_stats.get("accuracy").stat_list)
                     * np.array(eval_stats.get("num_eval_actions").stat_list)).sum(0)
                    / np.array(eval_stats.get("num_eval_actions").stat_list).sum(0))
            print(f"accuracy: {accuracy:.2%}, accuracy_without_no_op: {accuracy_without_no_op:.2%}")

        sys.exit(0)

    game_trajectories = load_dataset(args.dataset_path)

    max_len = get_max_len(eval_dataset, game_trajectories, len_func=get_bidding_length)
    print(f"max_len: {max_len}")
    clone_data_generator = bridgelearn.CloneDataGenerator(None, max_len, 1)  # type: ignore
    eval_batch = clone_data_generator.generate_eval_data(
        args.eval_batch_size,
        "cpu",  # Put batches in cpu here due to gpu usage.
        eval_dataset,
    )

    del clone_data_generator

    # Network and optimizer.
    agent = BridgeLSTMAgent(
        args.device,
        in_dim,
        args.hid_dim,
        out_dim,
        args.num_priv_mlp_layer,
        args.num_publ_mlp_layer,
        args.num_lstm_layer,
        args.activation,
        args.dropout,
        args.net
    )
    opt = torch.optim.Adam(agent.network.parameters(), lr=args.lr, eps=args.eps)

    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise ValueError(f"The model path {args.load_model} is not available.")
        checkpoint = torch.load(args.load_model)
        agent.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        agent.eval()
        print("Load trained model.")

        # Eval once for the trained model.
        eval_stats = evaluate(agent, eval_batch, args)
        print(f"eval_accuracy: {eval_stats.get('accuracy').mean()}, "
              f"eval_accuracy_without_no_op: {eval_stats.get('accuracy_without_no_op').mean()}, "
              f"eval_loss: {eval_stats.get('loss').mean()}")

        accuracy_without_no_op = (
                (np.array(eval_stats.get("accuracy_without_no_op").stat_list)
                 * np.array(eval_stats.get("num_eval_actions_without_no_op").stat_list)).sum(0)
                / np.array(eval_stats.get("num_eval_actions_without_no_op").stat_list).sum(0))

        accuracy = (
                (np.array(eval_stats.get("accuracy").stat_list)
                 * np.array(eval_stats.get("num_eval_actions").stat_list)).sum(0)
                / np.array(eval_stats.get("num_eval_actions").stat_list).sum(0))
        print(f"accuracy: {accuracy:.2%}, accuracy_without_no_op: {accuracy_without_no_op:.2%}")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    args.save_dir = mkdir_with_increment(args.save_dir)
    logger = Logger(os.path.join(args.save_dir, "log.txt"), auto_line_feed=True)
    logger.write(pprint.pformat(vars(args)))
    tachometer = Tachometer()

    data_gen, replay_buffer = create_data_generator(
        game_trajectories,
        max_len,
        args.num_threads,
        args.capacity,
        args.prefetch,
        args.seed,
    )

    data_gen.start_data_generation(bool(args.inf_loop), args.seed)

    while replay_buffer.size() < args.capacity:
        print(f"\r{replay_buffer.size()}/{args.capacity}", end="")
        time.sleep(1)

    # Main loop.
    stopwatch = Stopwatch()
    stats = MultiStats()
    saver = TopkSaver(args.save_dir, 5)
    tachometer.start()

    for epoch in range(args.num_epoch):
        stopwatch.reset()
        stats.reset()
        agent.train()
        train(
            agent,
            args.device,
            opt,
            replay_buffer,
            args.batch_size,
            args.epoch_len,
            args.max_grad_norm,
            args.no_op_actions_weight,
            stopwatch,
            stats,
        )

        print(f"Epoch {epoch}:")
        print(get_mem_usage())
        tachometer.lap(replay_buffer, args.epoch_len * args.batch_size, 1)
        stopwatch.summary()

        # Eval.
        agent.eval()
        with torch.no_grad():
            eval_stats = evaluate(agent, eval_batch, args)

        accuracy_without_no_op = (
                (np.array(eval_stats.get("accuracy_without_no_op").stat_list)
                 * np.array(eval_stats.get("num_eval_actions_without_no_op").stat_list)).sum(0)
                / np.array(eval_stats.get("num_eval_actions_without_no_op").stat_list).sum(0))

        accuracy = (
                (np.array(eval_stats.get("accuracy").stat_list)
                 * np.array(eval_stats.get("num_eval_actions").stat_list)).sum(0)
                / np.array(eval_stats.get("num_eval_actions").stat_list).sum(0))

        save_dict = {
            "model_state_dict": agent.state_dict(),
            "opt_state_dict": opt.state_dict(),
        }
        model_saved = saver.save(
            None, save_dict, eval_stats.get("accuracy_without_no_op").mean()
        )
        logger.write(
            f"Epoch {epoch}, loss: {stats.get('loss').mean():.4f}, "
            f"accuracy: {stats.get('accuracy').mean():.2%}, "
            f"accuracy_without_no_op: {stats.get('accuracy_without_no_op').mean():.2%}, "
            f"eval_accuracy: {accuracy:.2%}, "
            f"eval_accuracy_without_no_op: {accuracy_without_no_op:.2%}, "
            f"eval_loss: {eval_stats.get('loss').mean():.4f}, "
            f"model_saved: {model_saved}."
        )

    data_gen.terminate()
