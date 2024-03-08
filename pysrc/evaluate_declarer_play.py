"""
@author: qzz
@contact:q873264077@gmail.com
@file: evaluate_declarer_play.py
@time: 2024/03/03 12:18
"""

import os
import time
import multiprocessing as mp
import pickle
from typing import List, NamedTuple, Tuple
import socket

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import common_utils
from bluechip_bot import BlueChipBridgeBot
import set_path

set_path.append_sys_path()


import bridge
import torch
import rela
import bridgelearn
import bridgeplay
from create_bridge import BotFactory
from utils import load_dataset, extract_not_passed_out_trajectories
from evaluate_declarer_play_against_wbridge5 import DuplicateSaveItem


def construct_play_state_from_trajectory(
    trajectory: List[int], game: bridge.BridgeGame = bridge.default_game
) -> bridge.BridgeState:
    state = bridge.BridgeState(game)
    idx = 0
    while state.current_phase() != bridge.Phase.PLAY:
        uid = trajectory[idx]
        if state.is_chance_node():
            move = game.get_chance_outcome(uid)
        else:
            move = game.get_move(uid)
        state.apply_move(move)
        idx += 1
    assert state.current_phase() == bridge.Phase.PLAY
    return state


class Worker(mp.Process):
    def __init__(
        self,
        args: DictConfig,
        trajectories: List[List[int]],
        process_id: int,
        verbose: bool = False,
    ):
        super().__init__()
        self.args = args
        self._trajectories = trajectories
        self._process_id = process_id
        self._verbose = verbose

    def run(self):
        logger = common_utils.Logger(
            os.path.join(self.args.save_dir, f"log_{self._process_id}.txt"),
            auto_line_feed=True,
            verbose=False,
        )
        bot_factory: BotFactory = hydra.utils.instantiate(self.args.bot_factory)
        save_items = []

        defender_bots: List[bridgeplay.PlayBot] = [
            bot_factory.create_bot(
                self.args.defender, player_id=i, cmd_line=self.args.cmd_line
            )
            for i in range(bridge.NUM_PLAYERS)
        ]

        p1_bots = [
            bot_factory.create_bot(self.args.p1, player_id=i)
            for i in range(bridge.NUM_PLAYERS)
        ]

        p2_bots = [
            bot_factory.create_bot(self.args.p2, player_id=i)
            for i in range(bridge.NUM_PLAYERS)
        ]

        i_deal = 0

        print(f"process {self._process_id} start.")

        p1_execution_times = []
        p2_execution_times = []

        while i_deal < len(self._trajectories):
            for bot in defender_bots:
                bot.restart()
            for bot in p1_bots:
                bot.restart()

            cur_trajectory = self._trajectories[i_deal]

            state0 = construct_play_state_from_trajectory(cur_trajectory)

            # Play the game
            declarer = state0.get_contract().declarer
            while not state0.is_terminal():
                if state0.current_player() == declarer:
                    st = time.perf_counter()
                    move = p1_bots[declarer].step(state0)
                    ed = time.perf_counter()
                    p1_execution_times.append(ed - st)
                else:
                    move = defender_bots[state0.current_player()].step(state0)
                state0.apply_move(move)
            # print("State:\n", state)
            declarer_score0 = state0.scores()[declarer]
            # print("bot score: ", declarer_score)
            declarer_tricks0 = state0.num_declarer_tricks()

            state1 = construct_play_state_from_trajectory(cur_trajectory)

            for bot in defender_bots:
                bot.restart()
            for bot in p2_bots:
                bot.restart()
            # Play the game
            declarer = state1.get_contract().declarer
            while not state1.is_terminal():
                if state1.current_player() == declarer:
                    st = time.perf_counter()
                    move = p2_bots[declarer].step(state1)
                    ed = time.perf_counter()
                    p2_execution_times.append(ed - st)
                else:
                    move = defender_bots[state1.current_player()].step(state1)
                state1.apply_move(move)
            # print("State:\n", state)
            declarer_score1 = state1.scores()[declarer]
            # print("bot score: ", declarer_score)
            declarer_tricks1 = state1.num_declarer_tricks()
            i_deal += 1
            logger.write(
                f"Deal {i_deal},State0:\n{state0}\nstate1:\n{state1}\ntricks: {declarer_tricks0} : {declarer_tricks1}\nscores: {declarer_score0}: {declarer_score1}"
            )

            save_items.append(DuplicateSaveItem(state0, state1))
            print(f"process {self._process_id}, {i_deal}/{len(self._trajectories)}")

        with open(
            os.path.join(self.args.save_dir, f"items_{self._process_id}"), "wb"
        ) as fp:
            pickle.dump(save_items, fp, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(
            os.path.join(
                self.args.save_dir, f"p1_execution_times_{self._process_id}.npy"
            ),
            np.array(p1_execution_times),
        )
        np.save(
            os.path.join(
                self.args.save_dir, f"p2_execution_times_{self._process_id}.npy"
            ),
            np.array(p2_execution_times),
        )


@hydra.main("conf", "declarer_play", version_base="1.2")
def main(args: DictConfig):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    dataset = load_dataset(os.path.join(args.dataset_dir, args.dataset_name))

    dataset = extract_not_passed_out_trajectories(dataset)[:20]

    trajectories_per_process = common_utils.allocate_list_uniformly(
        dataset, args.num_processes
    )

    workers = [
        Worker(args, trajectories_per_process[i], i, False)
        for i in range(args.num_processes)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    # Gather results
    p1_execution_times = []
    p2_execution_times = []
    items = []
    logs = ""
    for i in range(args.num_processes):
        p1_execution_times.append(
            np.load(os.path.join(args.save_dir, f"p1_execution_times_{i}.npy"))
        )
        p2_execution_times.append(
            np.load(os.path.join(args.save_dir, f"p2_execution_times_{i}.npy"))
        )
        with open(os.path.join(args.save_dir, f"items_{i}"), "rb") as fp:
            items.extend(pickle.load(fp))
        with open(os.path.join(args.save_dir, f"log_{i}.txt"), "r") as f:
            logs += f.read() + "\n"

    p1_execution_times = np.concatenate(p1_execution_times)
    p2_execution_times = np.concatenate(p2_execution_times)
    # print(execution_times)
    np.save(os.path.join(args.save_dir, "p1_execution_times.npy"), p1_execution_times)
    np.save(os.path.join(args.save_dir, "p2_execution_times.npy"), p2_execution_times)
    with open(os.path.join(args.save_dir, "log.txt"), "w") as f:
        f.write(OmegaConf.to_yaml(args))
        f.write(logs)

    with open(os.path.join(args.save_dir, "items"), "wb") as fp:
        pickle.dump(items, fp, pickle.HIGHEST_PROTOCOL)

    print(f"Evaluation done!")
    print(f"p1:{args.p1} vs p2:{args.p2}, defender:{args.defender}")
    print(
        f"Execution time: {np.mean(p1_execution_times)} vs {np.mean(p2_execution_times)}"
    )

    p1_tricks = [item.state0.num_declarer_tricks() for item in items]
    p2_tricks = [item.state1.num_declarer_tricks() for item in items]
    print(f"Tricks: {np.mean(p1_tricks)} vs {np.mean(p2_tricks)}")

    p1_scores = [
        item.state0.scores()[item.state0.get_contract().declarer] for item in items
    ]
    p2_scores = [
        item.state1.scores()[item.state1.get_contract().declarer] for item in items
    ]
    print(f"Scores: {np.mean(p1_scores)} vs {np.mean(p2_scores)}")


if __name__ == "__main__":
    main()
