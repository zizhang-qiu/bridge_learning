import copy
import os.path
import pickle
import pprint
import sys
import time
from typing import Dict, Optional, Union, List, Tuple

import numpy as np
import torch
import hydra
import adan
from tqdm import trange

from agent import BridgeLSTMAgent
from set_path import append_sys_path
from common_utils import (get_mem_usage,
                          MultiStats,
                          get_avg_and_sem,
                          mkdir_with_increment,
                          Logger, TopkSaver, Stopwatch)
from utils import Tachometer

append_sys_path()
import bridge
import pyrela
import bridgelearn


def evaluate(args,
             runner0: pyrela.BatchRunner,
             runner1: pyrela.BatchRunner,
             num_threads: int,
             num_envs_per_thread: int,
             dataset_path: str,
             num_games: Optional[int] = None):
    st = time.perf_counter()
    with open(dataset_path, "rb") as fp:
        dataset = pickle.load(fp)
    if num_games is not None:
        cards = dataset["cards"][:num_games]
        ddts = dataset["ddts"][:num_games]
    else:
        cards = dataset["cards"]
        ddts = dataset["ddts"]
    num_games = len(cards)
    num_games_per_env = num_games // (num_threads * num_envs_per_thread)
    assert num_games_per_env * num_envs_per_thread * num_threads == num_games
    # bridge_dataset = bridgelearn.BridgeDataset(cards, ddts)
    env_actor_options = bridgelearn.EnvActorOptions()
    env_actor_options.eval = True

    context = pyrela.Context()
    all_env_actors = []
    for i_t in range(num_threads):
        env_actors = []
        for i_e in range(num_envs_per_thread):
            left = i_e * num_games_per_env + i_t * (num_envs_per_thread * num_games_per_env)
            right = left + num_games_per_env
            bridge_dataset = bridgelearn.BridgeDataset(
                cards[left:right],
                ddts[left:right]
            )
            env = create_env(args.dealer,
                             args.is_dealer_vulnerable,
                             args.is_non_dealer_vulnerable,
                             0,
                             bridge_dataset,
                             False)
            actors = [
                bridgelearn.BridgePublicLSTMActor(runner0, 0),
                bridgelearn.BridgePublicLSTMActor(runner1, 1),
                bridgelearn.BridgePublicLSTMActor(runner0, 2),
                bridgelearn.BridgePublicLSTMActor(runner1, 3)
            ]
            env_actor = bridgelearn.BridgeEnvActor(env, env_actor_options, actors)
            env_actors.append(env_actor)
        all_env_actors.extend(env_actors)

        thread_loop = bridgelearn.EnvActorThreadLoop(env_actors, num_games_per_env, i_t, False)
        context.push_thread_loop(thread_loop)

    context.start()
    context.join()

    rewards = []
    history_infos = []
    for ea in all_env_actors:
        env_history_rewards = ea.history_rewards()
        for r in env_history_rewards:
            rewards.append(r[0])
        history_infos.extend(ea.history_info())

    ed = time.perf_counter()
    return rewards, history_infos, ed - st


def create_env(dealer: int,
               is_dealer_vulnerable: bool,
               is_non_dealer_vulnerable: bool,
               seed: int,
               dataset: Optional[bridgelearn.BridgeDataset] = None,
               duplicate: bool = False) \
        -> Union[bridgelearn.BridgeEnv, bridgelearn.DuplicateEnv]:
    game_params = {
        "dealer": str(dealer),
        "is_dealer_vulnerable": str(is_dealer_vulnerable),
        "is_non_dealer_vulnerable": str(is_non_dealer_vulnerable),
        "seed": str(seed)
    }
    env_options = bridgelearn.BridgeEnvOptions()
    env_options.bidding_phase = True
    env_options.playing_phase = False
    env_options.verbose = False

    if duplicate:
        env = bridgelearn.DuplicateEnv(game_params, env_options)
    else:
        env = bridgelearn.BridgeEnv(game_params, env_options)

    if dataset is not None:
        env.set_bridge_dataset(dataset)

    return env


def create_agent_and_optimizer(args,
                               in_dim: int,
                               out_dim: int) \
        -> Tuple[BridgeLSTMAgent, pyrela.BatchRunner, pyrela.RNNPrioritizedReplay, torch.optim.Optimizer]:
    agent_cons = lambda: BridgeLSTMAgent(
        args.act_device,
        in_dim,
        args.hid_dim,
        out_dim,
        args.num_priv_mlp_layer,
        args.num_publ_mlp_layer,
        args.num_lstm_layer,
        args.activation,
        args.dropout,
        args.net,
        greedy=False
    )
    agent = agent_cons()
    # agent = agent.to(args.act_device)
    if args.sl_checkpoint:
        checkpoint = torch.load(args.sl_checkpoint, map_location=torch.device(args.act_device))
        agent.load_state_dict(checkpoint["model_state_dict"])
        print("load sl checkpoint")

    opt = torch.optim.Adam(agent.parameters(),
                           lr=args.lr)

    if args.rl_checkpoint:
        checkpoint = torch.load(args.rl_checkpoint, map_location=torch.device(args.act_device))
        agent.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        print("load rl checkpoint")

    runner = pyrela.BatchRunner(agent, args.act_device, 1000, ["act", "get_h0"])
    runner.start()

    buffer_seed = 3
    buffer = pyrela.RNNPrioritizedReplay(
        args.capacity,
        buffer_seed,
        args.alpha,
        args.beta,
        args.prefetch
    )

    return agent, runner, buffer, opt


def create_context(args,
                   runner: pyrela.BatchRunner,
                   buffer: pyrela.RNNPrioritizedReplay
                   ):
    assert args.train_dataset
    with open(args.train_dataset, "rb") as fp:
        dataset = pickle.load(fp)
    bridge_dataset = bridgelearn.BridgeDataset(dataset["cards"], dataset["ddts"])
    env_actor_options = bridgelearn.EnvActorOptions()
    env_actor_options.eval = False

    context = pyrela.Context()
    all_env_actors = []
    for i_t in trange(args.num_threads, desc="Creating threads"):
        env_actors = []
        for i_e in range(args.num_envs_per_thread):
            env = create_env(args.dealer,
                             args.is_dealer_vulnerable,
                             args.is_non_dealer_vulnerable,
                             0,
                             bridge_dataset,
                             args.duplicate)
            actors = [bridgelearn.BridgePublicLSTMActor(runner,
                                                        args.max_len,
                                                        args.gamma,
                                                        buffer,
                                                        idx) for idx in range(bridge.NUM_PLAYERS)]
            env_actor = bridgelearn.BridgeEnvActor(env, env_actor_options, actors)
            env_actors.append(env_actor)

        all_env_actors.extend(env_actors)
        thread_loop = bridgelearn.EnvActorThreadLoop(
            env_actors,
            -1,  # Infinite loop.
            i_t,
            False
        )
        context.push_thread_loop(thread_loop)

    return context, all_env_actors


@hydra.main("conf", "a2c.yaml", version_base="1.2")
def main(args):
    pprint.pprint(args)
    env = create_env(0, False, False, 1, None, False)
    in_dim = env.feature_size()
    out_dim = env.max_num_action() - bridge.NUM_CARDS
    print(in_dim, out_dim)
    del env

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_dir = mkdir_with_increment(args.save_dir)

    stats = MultiStats()
    logger = Logger(os.path.join(args.save_dir, "log.txt"), True, auto_line_feed=True)
    saver = TopkSaver(args.save_dir, 5)
    stopwatch = Stopwatch()
    tachometer = Tachometer()
    agent, runner, replay_buffer, opt = create_agent_and_optimizer(args, in_dim, out_dim)
    eval_rival_agent = BridgeLSTMAgent(
        args.act_device,
        in_dim,
        args.hid_dim,
        out_dim,
        args.num_priv_mlp_layer,
        args.num_publ_mlp_layer,
        args.num_lstm_layer,
        args.activation,
        args.dropout,
        args.net,
        greedy=True
    )

    eval_rival_agent.load_state_dict(agent.state_dict())
    eval_rival_runner = pyrela.BatchRunner(eval_rival_agent, args.act_device, 1000, ["act", "get_h0"])
    eval_rival_runner.start()
    # eval_agent = BridgeLSTMAgent(
    #     args.act_device,
    #     in_dim,
    #     args.hid_dim,
    #     out_dim,
    #     args.num_priv_mlp_layer,
    #     args.num_publ_mlp_layer,
    #     args.num_lstm_layer,
    #     args.activation,
    #     args.dropout,
    #     args.net,
    #     greedy=True
    # )
    # eval_agent.load_state_dict(agent.state_dict())
    # # eval_agent.eval()
    # # eval_agent.greedy = True
    # eval_runner = pyrela.BatchRunner(eval_agent, args.act_device, 1000, ["act", "get_h0"])
    # eval_runner.start()
    # rewards1, history_infos1, elapsed1 = evaluate(args, eval_runner, eval_rival_runner, args.num_eval_threads,
    #                                               args.num_eval_envs_per_thread,
    #                                               args.eval_dataset)
    # rewards2, history_infos2, elapsed2 = evaluate(args, eval_rival_runner, eval_runner, args.num_eval_threads,
    #                                               args.num_eval_envs_per_thread,
    #                                               args.eval_dataset)
    #
    # elapsed = elapsed1 + elapsed2
    # imps = [bridge.get_imp(int(score1), int(score2))
    #         for score1, score2 in zip(rewards1, rewards2)]
    # print(get_avg_and_sem(imps), elapsed)
    # for i in range(5):
    #     print(history_infos1[i], history_infos2[i], sep="\n")

    # for i, imp in enumerate(imps):
    #     if imp != 0:
    #         print(i)
    #         print(history_infos1[i], history_infos2[i], sep="\n")
    # sys.exit(0)

    context, all_env_actors = create_context(args, runner, replay_buffer)

    context.start()

    while (n := replay_buffer.size()) < args.burn_in:
        print(f"\rWarming up replay buffer, {n}/{args.burn_in}", end="")
    print()

    for i_epoch in range(args.num_epochs):
        print(f"Epoch {i_epoch}.")
        print(get_mem_usage())
        stats.reset()
        stopwatch.reset()
        tachometer.start()
        for i_batch in trange(args.epoch_len):
            num_update = i_batch + i_epoch * args.epoch_len
            if num_update % args.synq_freq == 0:
                runner.update_model(agent)
            torch.cuda.synchronize()
            stopwatch.time("Synchronize model")

            batch, weight = replay_buffer.sample(args.batch_size, args.train_device)
            stopwatch.time("Sample data")
            p_loss, v_loss, priority = agent.compute_loss_and_priority(batch,
                                                                       args.clip_eps,
                                                                       args.entropy_ratio,
                                                                       args.value_loss_weight)
            stopwatch.time("forward & backward")

            weighted_loss = ((p_loss + v_loss) * weight).mean()
            weighted_loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

            opt.step()
            opt.zero_grad()
            stopwatch.time("update model")

            replay_buffer.update_priority(priority)
            stopwatch.time("update priority")

            stats.feed("g_norm", g_norm.item())
            stats.feed("p_loss", p_loss.mean().item())
            stats.feed("v_loss", v_loss.mean().item())
        for stat in stats.stats.values():
            print(stat.summary())
        tachometer.lap(replay_buffer, args.epoch_len * args.batch_size, 1)
        stopwatch.summary()

        # Evaluate.
        context.pause()
        eval_agent = BridgeLSTMAgent(
            args.act_device,
            in_dim,
            args.hid_dim,
            out_dim,
            args.num_priv_mlp_layer,
            args.num_publ_mlp_layer,
            args.num_lstm_layer,
            args.activation,
            args.dropout,
            args.net,
            greedy=True
        )
        eval_agent.load_state_dict(agent.state_dict())
        eval_runner = pyrela.BatchRunner(eval_agent, args.act_device, 1000, ["act", "get_h0"])
        eval_runner.start()
        rewards1, history_infos1, elapsed1 = evaluate(args, eval_runner, eval_rival_runner, args.num_eval_threads,
                                                      args.num_eval_envs_per_thread,
                                                      args.eval_dataset)
        rewards2, history_infos2, elapsed2 = evaluate(args, eval_rival_runner, eval_runner, args.num_eval_threads,
                                                      args.num_eval_envs_per_thread,
                                                      args.eval_dataset)

        elapsed = elapsed1 + elapsed2
        imps = [bridge.get_imp(int(score1), int(score2)) for score1, score2 in zip(rewards1, rewards2)]
        checkpoint = {"model_state_dict": agent.state_dict(),
                      "opt_state_dict": opt.state_dict()}
        model_saved = saver.save(None, checkpoint, np.mean(imps).item(), False)

        logger.write(
            f"Epoch {i_epoch}, imps: {get_avg_and_sem(imps)}, elapsed: {elapsed:.2f}, model_saved:{model_saved}")

        context.resume()


if __name__ == '__main__':
    main()
