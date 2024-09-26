import os.path
import pickle
import pprint
import random
import time
from collections import deque
from typing import Optional, Union, List, Tuple, Callable, Any

import adan
import hydra
import numpy as np
import omegaconf.dictconfig
import torch
from tqdm import trange

from agent import BridgeFFWDAgent
from common_utils import (
    get_mem_usage,
    MultiStats,
    get_avg_and_sem,
    mkdir_with_increment,
    Logger,
    TopkSaver,
    Stopwatch,
)
from set_path import append_sys_path
from utils import Tachometer

append_sys_path()
import bridge
import pyrela
import bridgelearn


def evaluate_once(
    args,
    actor0_cons: Callable[[int], bridgelearn.Actor],
    actor1_cons: Callable[[int], bridgelearn.Actor],
    num_threads: int,
    num_envs_per_thread: int,
    dataset_path: str,
    num_games: Optional[int] = None,
) -> Tuple[List[float], List[str], float]:
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
    bridge_dataset = bridgelearn.BridgeDataset(cards, ddts)
    env_actor_options = bridgelearn.EnvActorOptions()
    env_actor_options.eval = True

    context = pyrela.Context()
    all_env_actors = []
    for i_t in range(num_threads):
        env_actors = []
        for i_e in range(num_envs_per_thread):
            env = create_env(
                args.dealer,
                args.is_dealer_vulnerable,
                args.is_non_dealer_vulnerable,
                0,
                args.encoder,
                bridge_dataset,
                True,
            )
            actors = [actor0_cons(0), actor1_cons(1), actor0_cons(2), actor1_cons(3)]
            env_actor = bridgelearn.BridgeEnvActor(env, env_actor_options, actors)
            env_actors.append(env_actor)
        all_env_actors.extend(env_actors)

        thread_loop = bridgelearn.EnvActorThreadLoop(
            env_actors, num_games_per_env, i_t, False
        )
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


def evaluate(
    agent: BridgeFFWDAgent,
    eval_rival_runner: pyrela.BatchRunner,
    args,
    num_display: int = 0,
) -> Tuple[List[float], float]:
    eval_agent = agent.clone(args.act_device, overwrite={"greedy": True})
    eval_agent.eval()
    eval_runner = pyrela.BatchRunner(
        eval_agent, args.act_device, 1000, ["act", "get_h0"]
    )
    eval_runner.start()
    eval_actor_cons = lambda player_idx: bridgelearn.BridgeFFWDActor(
        eval_runner, player_idx
    )
    eval_rival_actor_cons = lambda player_idx: bridgelearn.BridgeFFWDActor(
        eval_rival_runner, player_idx
    )
    rewards1, history_infos1, elapsed1 = evaluate_once(
        args,
        eval_actor_cons,
        eval_rival_actor_cons,
        args.num_eval_threads,
        args.num_eval_envs_per_thread,
        args.eval_dataset,
        args.num_eval_games,
    )

    return rewards1, elapsed1


def evaluate_with_all_pass_actor(agent, args, num_display: int = 0):
    eval_agent = agent.clone(args.act_device, overwrite={"greedy": True})
    eval_agent.eval()
    eval_runner = pyrela.BatchRunner(eval_agent, args.act_device, 1000, ["act"])
    eval_runner.start()
    eval_actor_cons = lambda player_idx: bridgelearn.BridgeFFWDActor(
        eval_runner, player_idx
    )
    all_pass_actor_cons = lambda player_idx: bridgelearn.AllPassActor(player_idx)
    rewards1, history_infos1, elapsed1 = evaluate_once(
        args,
        eval_actor_cons,
        all_pass_actor_cons,
        args.num_eval_threads,
        args.num_eval_envs_per_thread,
        args.eval_dataset,
        args.num_eval_games,
    )
    elapsed = elapsed1

    return rewards1, elapsed


def create_env(
    dealer: int,
    is_dealer_vulnerable: bool,
    is_non_dealer_vulnerable: bool,
    seed: int,
    encoder: str,
    dataset: Optional[bridgelearn.BridgeDataset] = None,
    duplicate: bool = False,
) -> Union[bridgelearn.BridgeEnv, bridgelearn.DuplicateEnv]:
    game_params = {
        "dealer": str(dealer),
        "is_dealer_vulnerable": str(is_dealer_vulnerable),
        "is_non_dealer_vulnerable": str(is_non_dealer_vulnerable),
        "seed": str(seed),
    }
    env_options = bridgelearn.BridgeEnvOptions()
    env_options.bidding_phase = True
    env_options.playing_phase = False
    env_options.encoder = encoder
    env_options.verbose = False
    env_options.max_len = -1

    if duplicate:
        env = bridgelearn.DuplicateEnv(game_params, env_options)
    else:
        env = bridgelearn.BridgeEnv(game_params, env_options)

    if dataset is not None:
        env.set_bridge_dataset(dataset)

    return env


def create_sl_agent(args, p_in_dim: int, v_in_dim: int, out_dim: int):
    assert args.sl_checkpoint
    agent_cons = lambda: BridgeFFWDAgent(
        args.act_device,
        p_in_dim,
        v_in_dim,
        args.p_hid_dim,
        args.v_hid_dim,
        out_dim,
        args.num_p_mlp_layer,
        args.num_v_mlp_layer,
        args.p_activation,
        args.v_activation,
        args.dropout,
        args.net,
        greedy=True,
    )
    agent = agent_cons()
    checkpoint = torch.load(args.sl_checkpoint)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return agent.to(args.act_device)


def create_agent_and_optimizer(
    args, p_in_dim: int, v_in_dim: int, out_dim: int
) -> Tuple[
    BridgeFFWDAgent,
    BridgeFFWDAgent,
    pyrela.FFPrioritizedReplay,
    torch.optim.Optimizer,
]:
    agent_cons = lambda: BridgeFFWDAgent(
        args.act_device,
        p_in_dim,
        v_in_dim,
        args.p_hid_dim,
        args.v_hid_dim,
        out_dim,
        args.num_p_mlp_layer,
        args.num_v_mlp_layer,
        args.p_activation,
        args.v_activation,
        args.dropout,
        args.net,
        greedy=False,
        uniform_priority=args.uniform_priority,
        reuse_value_in_priority=args.reuse_value_in_priority,
    )
    agent = agent_cons()
    agent = agent.to(args.act_device)
    if args.sl_checkpoint and args.sl_init and not args.rl_checkpoint:
        checkpoint = torch.load(
            args.sl_checkpoint, map_location=torch.device(args.act_device)
        )
        agent.load_state_dict(checkpoint["model_state_dict"])
        print("Load sl checkpoint.")

    # opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    opt = adan.Adan(agent.parameters(), lr=args.lr)

    if args.rl_checkpoint:
        checkpoint = torch.load(
            args.rl_checkpoint, map_location=torch.device(args.act_device)
        )
        agent.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["opt_state_dict"])
        print("Load rl checkpoint.")

    oppo_agent = agent.clone(args.act_device)
    oppo_agent.eval()

    buffer_seed = int(time.perf_counter()) % 1234567
    buffer = pyrela.FFPrioritizedReplay(
        args.capacity, buffer_seed, args.alpha, args.beta, args.prefetch
    )

    return agent, oppo_agent, buffer, opt


def create_context(
    args,
    runner: pyrela.BatchRunner,
    oppo_runner: pyrela.BatchRunner,
    buffer: pyrela.FFPrioritizedReplay,
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
            env0 = create_env(
                args.dealer,
                args.is_dealer_vulnerable,
                args.is_non_dealer_vulnerable,
                0,
                args.encoder,
                bridge_dataset,
                args.duplicate,
            )
            actors0 = [
                bridgelearn.BridgeFFWDActor(runner, args.gamma, buffer, 0),
                bridgelearn.BridgeFFWDActor(oppo_runner, 1),
                bridgelearn.BridgeFFWDActor(runner, args.gamma, buffer, 2),
                bridgelearn.BridgeFFWDActor(oppo_runner, 3),
            ]
            env_actor0 = bridgelearn.BridgeEnvActor(env0, env_actor_options, actors0)
            env_actors.append(env_actor0)

            env1 = create_env(
                args.dealer,
                args.is_dealer_vulnerable,
                args.is_non_dealer_vulnerable,
                0,
                args.encoder,
                bridge_dataset,
                args.duplicate,
            )
            actors1 = [
                bridgelearn.BridgeFFWDActor(oppo_runner, 0),
                bridgelearn.BridgeFFWDActor(runner, args.gamma, buffer, 1),
                bridgelearn.BridgeFFWDActor(oppo_runner, 2),
                bridgelearn.BridgeFFWDActor(runner, args.gamma, buffer, 3),
            ]
            env_actor1 = bridgelearn.BridgeEnvActor(env1, env_actor_options, actors1)

            env_actors.append(env_actor1)

        all_env_actors.extend(env_actors)
        thread_loop = bridgelearn.EnvActorThreadLoop(
            env_actors, -1, i_t, False  # Infinite loop.
        )
        context.push_thread_loop(thread_loop)

    return context, all_env_actors


class ModelPool:
    def __init__(self, capacity: int):
        self.models = deque([], capacity)

    def add(self, model: Any) -> None:
        self.models.append(model)

    def sample(self):
        idx = random.randint(0, len(self.models) - 1)
        return self.models[idx]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, item: int):
        return self.models[item]


@hydra.main("conf", "ffwd_a2c.yaml", version_base="1.2")
def main(args: omegaconf.dictconfig.DictConfig):
    os.environ["OMP_NUM_THREADS"] = "1"

    torch.set_printoptions(threshold=100000000000)
    # pprint.pprint(omegaconf.OmegaConf.to_container(args))
    env = create_env(0, False, False, 1, args.encoder, None, args.duplicate)
    perf_size, priv_size, publ_size = env.feature_size()
    out_dim = env.max_num_action() - bridge.NUM_CARDS - 1
    print(f"in_dim: {perf_size, priv_size}, out_dim: {out_dim}.")
    del env

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_dir = mkdir_with_increment(args.save_dir)

    global_stats = MultiStats()
    stats = MultiStats()
    logger = Logger(os.path.join(args.save_dir, "train.log"), True, auto_line_feed=True)
    logger.write(pprint.pformat(omegaconf.OmegaConf.to_container(args)))
    saver = TopkSaver(args.save_dir, 5)
    stopwatch = Stopwatch()
    tachometer = Tachometer()
    agent, oppo_agent, replay_buffer, opt = create_agent_and_optimizer(
        args, priv_size, perf_size, out_dim
    )
    runner = pyrela.BatchRunner(
        agent, args.act_device, 1000, ["act", "compute_priority"]
    )
    runner.start()
    oppo_runner = pyrela.BatchRunner(
        oppo_agent, args.act_device, 1000, ["act", "compute_priority"]
    )
    oppo_runner.start()
    eval_oppo_agent = create_sl_agent(args, priv_size, perf_size, out_dim)
    eval_oppo_runner = pyrela.BatchRunner(
        eval_oppo_agent, args.act_device, 1000, ["act", "get_h0"]
    )
    eval_oppo_runner.start()

    model_pool = ModelPool(args.model_pool_capacity)
    model_pool.add(agent.clone(args.act_device))
    # print("Evaluating with sl actor...")
    # imps, elapsed = evaluate(agent, eval_oppo_runner, args)
    # imps = np.round(np.array(imps) * 24)
    # print(imps, get_avg_and_sem(imps), elapsed)
    #
    # print("Evaluating with all pass actor...")
    # imps, elapsed = evaluate_with_all_pass_actor(agent, args)
    # imps = np.round(np.array(imps) * 24)
    # print(imps, get_avg_and_sem(imps), elapsed)

    context, all_env_actors = create_context(args, runner, oppo_runner, replay_buffer)

    context.start()

    while (n := replay_buffer.size()) < args.burn_in:
        print(f"\rWarming up replay buffer, {n}/{args.burn_in}", end="")
        time.sleep(0.5)
    print()
    # sys.exit(0)

    for i_epoch in range(args.num_epochs):
        print(f"Epoch {i_epoch}.")
        print(get_mem_usage())
        stats.reset()
        stopwatch.reset()
        tachometer.start()
        for i_batch in range(args.epoch_len):
            num_update = i_batch + i_epoch * args.epoch_len
            if num_update % args.synq_freq == 0:
                runner.update_model(agent)
                # print(f"num_update={num_update}, model updated.")
            if args.oppo_synq_freq > 0 and num_update % args.oppo_synq_freq == 0:
                oppo_agent_sample = model_pool.sample()
                oppo_runner.update_model(oppo_agent_sample)
            torch.cuda.synchronize()
            stopwatch.time("Synchronize model")

            batch, weight = replay_buffer.sample(args.batch_size, args.train_device)
            # print(weight)
            # for k, v in batch.obs.items():
            #     print(k, v)
            # print(batch.reward)
            # print(batch.reward * 24)
            # print(batch.seq_len)
            stopwatch.time("Sample data")
            p_loss, v_loss, priority = agent.compute_loss_and_priority(
                batch,
                args.clip_eps,
                args.entropy_ratio,
                args.value_loss_weight,
            )
            # print(priority)
            stopwatch.time("Forward & backward")
            # print(p_loss, v_loss, priority, sep="\n")

            loss = ((p_loss + v_loss) * weight).mean()
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.parameters(), args.max_grad_norm
            )

            opt.step()
            opt.zero_grad()
            stopwatch.time("Update model")

            replay_buffer.update_priority(priority)
            stopwatch.time("Update priority")

            stats.feed("g_norm", g_norm.item())
            stats.feed("p_loss", p_loss.mean().item())
            stats.feed("v_loss", v_loss.mean().item())
            stats.feed("loss", loss.detach().item())

            # input()
        for stat in stats.stats.values():
            print(stat.summary())
        tachometer.lap(replay_buffer, args.epoch_len * args.batch_size, 1)
        stopwatch.summary()

        # Save to model pool
        model_pool.add(agent.clone(args.act_device))

        # Evaluate.
        # context.pause()
        with torch.no_grad():
            print("Evaluating with all pass actor...")
            imps, elapsed = evaluate_with_all_pass_actor(agent, args, 3)
            imps = np.round(np.array(imps) * 24)
            print(imps, get_avg_and_sem(imps), elapsed)

            print("Evaluating with sl actor...")
            imps, elapsed = evaluate(agent, eval_oppo_runner, args, 3)
            imps = np.round(np.array(imps) * 24)
            print(imps, get_avg_and_sem(imps), elapsed)

        avg_imp, sem_imp = get_avg_and_sem(imps)
        global_stats.feed("avg_imp", avg_imp)
        global_stats.feed("sem_imp", sem_imp)
        global_stats.feed("elapsed", elapsed)
        global_stats.save_all(args.save_dir, plot=True)
        checkpoint = {
            "model_state_dict": agent.state_dict(),
            "opt_state_dict": opt.state_dict(),
        }
        model_saved = saver.save(None, checkpoint, avg_imp, False)

        logger.write(
            f"Epoch {i_epoch}, imps: {avg_imp:.4f} \u00B1 {sem_imp:.4f}, elapsed: {elapsed:.2f}, model_saved:{model_saved}"
        )

    #     # context.resume()


if __name__ == "__main__":
    main()
