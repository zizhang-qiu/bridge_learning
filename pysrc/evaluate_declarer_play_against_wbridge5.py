import os
import multiprocessing as mp
from typing import List, NamedTuple
import socket

import hydra
import numpy as np
from omegaconf import DictConfig
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

def signal(lhs:int, rhs: int) -> int:
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
        process_id:int,
        queue:mp.SimpleQueue,
        verbose: bool = False,
    ):
        super().__init__()
        self.args = args
        self._trajectories = trajectories
        self._process_id = process_id
        self._verbose = verbose
        self.q = queue

    def run(self):
        bot_factory: BotFactory = hydra.utils.instantiate(self.args.bot_factory)

        wbridge_bots: List[BlueChipBridgeBot] = [
            bot_factory.create_bot("bluechip", player_id=i, cmd_line=self.args.cmd_line)
            for i in range(bridge.NUM_PLAYERS)
        ]

        declarer_bots = [
            bot_factory.create_bot(self.args.declarer, player_id=i)
            for i in range(bridge.NUM_PLAYERS)
        ]

        i_deal = 0

        print(f"process {self._process_id} start.")

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

            declarer = state.get_contract().declarer
            while not state.is_terminal():
                if state.current_player() == declarer:
                    move = declarer_bots[declarer].step(state)
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
            wbridge5_declarer_tricks = original_state.num_declarer_tricks();
            if self.q is not None:
                self.q.put((declarer_tricks - wbridge5_declarer_tricks,
                            declarer_score-wbridge5_declarer_score))

            i_deal += 1
            print(f"process {self._process_id}, {i_deal}/{len(self._trajectories)}")


@hydra.main("conf", "declarer_play_against_wbridge5", version_base="1.2")
def main(args: DictConfig):
    dataset = load_dataset(os.path.join(args.dataset_dir, args.dataset_name))

    dataset = extract_not_passed_out_trajectories(dataset)[:1000]

    # print(len(dataset))

    trajectories_per_process = common_utils.allocate_list_uniformly(
        dataset, args.num_processes
    )
    q = mp.SimpleQueue()
    workers = [Worker(args, trajectories_per_process[i], i, q, False) for i in range(args.num_processes)]

    for w in  workers:
        w.start()

    for w in workers:
        w.join()

    trick_results = []
    score_results = []

    while not q.empty():
        item = q.get()
        trick_results.append(item[0])
        score_results.append(item[1])

    print(np.mean(trick_results))
    print(np.mean(score_results))

if __name__ == "__main__":
    main()
