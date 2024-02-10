"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: validate_wbridge5_trajectories.py
@time: 2024/2/9 9:44
"""
import os
from typing import List
import multiprocessing as mp

from set_path import append_sys_path

append_sys_path()

import bridge
import torch
import rela
import bridgelearn
import bridgeplay

from utils import load_dataset
from create_bridge import create_params, create_bridge_game
from bluechip_bot import BlueChipBridgeBot
from wbridge5_client import WBridge5Client, Controller
import common_utils

bot_cmd = "D:/wbridge5/Wbridge5.exe Autoconnect {port}"
timeout_secs = 120

params = create_params(seed=23)
game = create_bridge_game(params)


def controller_factory() -> Controller:
    client = WBridge5Client(bot_cmd, timeout_secs)
    client.start()
    return client


def create_wbridge_bots() -> List[BlueChipBridgeBot]:
    bots = [
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.NORTH,
            controller_factory=controller_factory
        ),
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.EAST,
            controller_factory=controller_factory
        ),
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.SOUTH,
            controller_factory=controller_factory
        ),
        BlueChipBridgeBot(
            game=game,
            player_id=bridge.Seat.WEST,
            controller_factory=controller_factory
        ),
    ]

    return bots


class ValidateWorker(mp.Process):
    def __init__(self, process_id: int, trajectories: List[List[int]], verbose: bool = False):
        super().__init__()
        self._process_id = process_id
        self._trajectories = trajectories
        self._verbose = verbose

    def run(self):
        bots = create_wbridge_bots()
        num_match = 0
        for i, trajectory in enumerate(self._trajectories):
            for bot in bots:
                bot.restart()
            state = bridgeplay.construct_state_from_deal(trajectory[:bridge.NUM_CARDS], game)
            idx = bridge.NUM_CARDS
            match = True
            while state.current_phase() == bridge.Phase.AUCTION:
                uid = bots[state.current_player()].step(state)
                state.apply_move(game.get_move(uid))
                if uid != trajectory[idx]:
                    # Print the state to see.
                    print(
                        f"At state {state}\n"
                        f"move in dataset: {game.get_move(trajectory[idx])}, "
                        f"move by wbridge5: {game.get_move(uid)}")
                    match = False
                    continue
                idx += 1

            num_match += int(match)
            print(f"Process {self._process_id}: {num_match}/{i + 1}, total {len(self._trajectories)}")


def main():
    dataset_dir = r"D:\Projects\bridge_research\expert"
    test_dataset = load_dataset(os.path.join(dataset_dir, "test.txt"))

    num_processes = mp.cpu_count()
    print("Num processes: ", num_processes)

    trajectories_per_process = common_utils.allocate_list_uniformly(test_dataset, num_processes)
    workers = []
    for i in range(num_processes):
        worker = ValidateWorker(i, trajectories_per_process[i], verbose=True)
        workers.append(worker)

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == '__main__':
    main()
