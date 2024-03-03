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


class DuplicateSaveItem:
    state0: bridge.BridgeState
    state1: bridge.BridgeState
    """An item to save duplicate results"""

    def __init__(self, state0: bridge.BridgeState, state1: bridge.BridgeState) -> None:
        self.state0 = state0
        self.state1 = state1

    def get_declarer_tricks(self) -> Tuple[int, int]:
        return (self.state0.num_declarer_tricks(), self.state1.num_declarer_tricks())

    def get_declarer_scores(self) -> Tuple[int, int]:
        return (
            self.state0.scores()[self.state0.get_contract().declarer],
            self.state1.scores()[self.state1.get_contract().declarer],
        )


def signal(lhs: int, rhs: int) -> int:
    if lhs > rhs:
        return 1
    if lhs < rhs:
        return -1
    return 0


class Worker(mp.Process):
    def __init__(
        self,
        args: DictConfig,
        trajectories: List[List[int]],
        process_id: int,
        queue: mp.SimpleQueue,
        verbose: bool = False,
    ):
        super().__init__()
        self.args = args
        self._trajectories = trajectories
        self._process_id = process_id
        self._verbose = verbose
        self.q = queue

    def run(self):
        logger = common_utils.Logger(
            os.path.join(self.args.save_dir, f"log_{self._process_id}.txt"),
            auto_line_feed=True,
            verbose=False,
        )
        bot_factory: BotFactory = hydra.utils.instantiate(self.args.bot_factory)
        save_items = []

        wbridge_bots: List[BlueChipBridgeBot] = [
            bot_factory.create_bot("bluechip", player_id=i, cmd_line=self.args.cmd_line)
            for i in range(bridge.NUM_PLAYERS)
        ]  # type: ignore

        declarer_bots = [
            bot_factory.create_bot(self.args.declarer, player_id=i)
            for i in range(bridge.NUM_PLAYERS)
        ]

        i_deal = 0

        print(f"process {self._process_id} start.")

        execution_times = []

        while i_deal < len(self._trajectories):
            for bot in wbridge_bots:
                bot.restart()
            for bot in declarer_bots:
                bot.restart()

            cur_trajectory = self._trajectories[i_deal]

            # Check if Wbridge5 make calls identically to trajectory.
            try:
                move_idx = bridge.NUM_CARDS
                state = bridgeplay.construct_state_from_deal(
                    cur_trajectory[: bridge.NUM_CARDS], bridge.default_game
                )
                match = True
                while state.current_phase() == bridge.Phase.AUCTION:
                    uid = wbridge_bots[state.current_player()]._step(state)
                    state.apply_move(bridge.default_game.get_move(uid))
                    if uid != cur_trajectory[move_idx]:
                        i_deal += 1
                        match = False
                        break
                    move_idx += 1
                if not match:
                    continue

            except socket.timeout as e:
                continue

            # Play the game
            declarer = state.get_contract().declarer
            wbridge_bots[declarer].restart()
            wbridge_bots[bridge.partner(declarer)].restart()
            while not state.is_terminal():
                if state.current_player() == declarer:
                    st = time.perf_counter()
                    move = declarer_bots[declarer].step(state)
                    ed = time.perf_counter()
                    execution_times.append(ed - st)
                else:
                    move = wbridge_bots[state.current_player()].step(state)
                state.apply_move(move)
            # print("State:\n", state)
            declarer_score = state.scores()[declarer]
            # print("bot score: ", declarer_score)
            declarer_tricks = state.num_declarer_tricks()

            # Get the result of WBridge5.
            original_state = bridgeplay.construct_state_from_trajectory(
                cur_trajectory, bridge.default_game
            )
            # print("Original state:\n", original_state)
            wbridge5_declarer_score = original_state.scores()[
                original_state.get_contract().declarer
            ]
            # print("wbridge5 score: ", wbridge5_declarer_score)
            wbridge5_declarer_tricks = original_state.num_declarer_tricks()

            i_deal += 1
            logger.write(
                f"Deal {i_deal},State:\n{state}\nOriginal state:\n{original_state}\ntricks: {declarer_tricks} : {wbridge5_declarer_tricks}\nscores: {declarer_score}: {wbridge5_declarer_score}"
            )

            save_items.append(DuplicateSaveItem(state, original_state))
            print(f"process {self._process_id}, {i_deal}/{len(self._trajectories)}")

        with open(
            os.path.join(self.args.save_dir, f"items_{self._process_id}"), "wb"
        ) as fp:
            pickle.dump(save_items, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
        np.save(os.path.join(self.args.save_dir, f"execution_times_{self._process_id}.npy"), np.array(execution_times))


@hydra.main("conf", "declarer_play_against_wbridge5", version_base="1.2")
def main(args: DictConfig):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    dataset = load_dataset(os.path.join(args.dataset_dir, args.dataset_name))

    dataset = extract_not_passed_out_trajectories(dataset)[:500]

    # print(len(dataset))

    trajectories_per_process = common_utils.allocate_list_uniformly(
        dataset, args.num_processes
    )
    
    
    q = mp.SimpleQueue()
    workers = [
        Worker(args, trajectories_per_process[i], i, q, False)
        for i in range(args.num_processes)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    # Gather results
    execution_times = []
    items = []
    logs = ""
    for i in range(args.num_processes):
        execution_times.append(np.load(os.path.join(args.save_dir, f"execution_times_{i}.npy")))
        with open(os.path.join(args.save_dir, f"items_{i}"), "rb") as fp:
            items.extend(pickle.load(fp))
        with open(os.path.join(args.save_dir, f"log_{i}.txt"), "r") as f:
            logs += f.read() + "\n"
    
    execution_times = np.concatenate(execution_times)
    # print(execution_times)
    np.save(os.path.join(args.save_dir, "execution_times.npy"), execution_times)
    with open(os.path.join(args.save_dir, "log.txt"), "w") as f:
        f.write(OmegaConf.to_yaml(args))
        f.write(logs)
    
    with open(os.path.join(args.save_dir, "items"), "wb") as fp:
        pickle.dump(items, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
    # m=2 w=20 -0.175 -29.95
    # m=2, w=40 -0.2 -30.15
    # m=3, w=20 -0.125 -25.55
    # m=3, w=40 -0.205 -26
