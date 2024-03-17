import pickle
from typing import List, Tuple
import hydra
import os
import multiprocessing as mp
import omegaconf
import torch
import time
import numpy as np

from evaluate_declarer_play_against_wbridge5 import DuplicateSaveItem
import set_path

set_path.append_sys_path()
import bridge
import rela
import bridgeplay
import common_utils

from create_bridge import BotFactory
from utils import load_dataset, extract_not_passed_out_trajectories


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


def construct_bidding_state(
    trajectory: List[int], game: bridge.BridgeGame = bridge.default_game
) -> bridge.BridgeState:
    state = bridge.BridgeState(game)
    for i in range(bridge.NUM_CARDS):
        uid = trajectory[i]
        chance = game.get_chance_outcome(uid)
        state.apply_move(chance)

    return state


def get_declarer_from_trajectory(
    trajectory: List[int], game: bridge.BridgeGame = bridge.default_game
):
    state = construct_play_state_from_trajectory(trajectory, game)
    return state.get_contract().declarer


def restart_all_bots(bots: List[bridgeplay.PlayBot]):
    for bot in bots:
        bot.restart()


class Worker(mp.Process):
    def __init__(
        self, args: omegaconf.OmegaConf, trajectories: List[List[int]], save_dir : str, process_id: int
    ):
        super().__init__()
        self.args = args
        self.trajectories = trajectories
        self.save_dir = save_dir
        self._process_id = process_id

    def run_bidding(
        self,
        cur_trajectory: List[int],
        declarer_bots: List[bridgeplay.PlayBot],
        defender_bots: List[bridgeplay.PlayBot],
    ) -> Tuple[bridge.BridgeState, bool]:
        declarer = get_declarer_from_trajectory(cur_trajectory)
        declarer_side_players = [declarer, bridge.partnership(declarer)]

        state = construct_bidding_state(cur_trajectory)

        while state.current_phase() == bridge.Phase.AUCTION:
            current_player = state.current_player()
            if current_player in declarer_side_players:
                move = declarer_bots[current_player].step(state)
            else:
                move = defender_bots[current_player].step(state)
            state.apply_move(move)

        if state.uid_history() != cur_trajectory[: -bridge.NUM_CARDS]:
            return state, False
        else:
            return state, True

    def run_playing(
        self,
        state: bridge.BridgeState,
        declarer_bots: List[bridgeplay.PlayBot],
        defender_bots: List[bridgeplay.PlayBot],
    ) -> Tuple[bridge.BridgeState, List[float], List[float]]:
        declarer = state.get_contract().declarer
        declarer_side_players = [declarer, bridge.partnership(declarer)]
        declarer_execution_times = []
        defender_execution_times = []
        while not state.is_terminal():
            current_player = state.current_player()
            if current_player in declarer_side_players:
                st = time.perf_counter()
                move = declarer_bots[current_player].step(state)
                ed = time.perf_counter()
                declarer_execution_times.append(ed - st)
            else:
                st = time.perf_counter()
                move = defender_bots[current_player].step(state)
                ed = time.perf_counter()
                defender_execution_times.append(ed - st)
            state.apply_move(move)

        return state, declarer_execution_times, defender_execution_times

    def run(self) -> None:
        # Create bot factory
        bot_factory: BotFactory = hydra.utils.instantiate(self.args.bot_factory)

        logger = common_utils.Logger(
            os.path.join(self.save_dir, f"log_{self._process_id}.txt"),
            auto_line_feed=True,
            verbose=False,
        )

        bidding_bot = bridgeplay.TrajectoryBiddingBot(
            self.trajectories, bridge.default_game
        )

        defender_bots: List[bridgeplay.PlayBot] = [
            bot_factory.create_bot(
                self.args.defender, player_id=i, cmd_line=self.args.cmd_line
            )
            for i in range(bridge.NUM_PLAYERS)
        ]
        for bot in defender_bots:
            if hasattr(bot, "set_bidding_bot"):
                bot.set_bidding_bot(bidding_bot)

        p1_bots = [
            bot_factory.create_bot(self.args.p1, player_id=i, **self.args.p1_args)
            for i in range(bridge.NUM_PLAYERS)
        ]
        for bot in p1_bots:
            if hasattr(bot, "set_bidding_bot"):
                bot.set_bidding_bot(bidding_bot)

        p2_bots = [
            bot_factory.create_bot(self.args.p2, player_id=i, **self.args.p2_args)
            for i in range(bridge.NUM_PLAYERS)
        ]
        for bot in p2_bots:
            if hasattr(bot, "set_bidding_bot"):
                bot.set_bidding_bot(bidding_bot)

        p1_times = []
        p2_times = []
        defender_times = []
        save_items = []

        deal_idx = 0
        while deal_idx < len(self.trajectories):
            # restart all the bots
            restart_all_bots(p1_bots)
            restart_all_bots(p2_bots)
            restart_all_bots(defender_bots)

            cur_trajectory = self.trajectories[deal_idx]
            # Open table.
            state0, available = self.run_bidding(cur_trajectory, p1_bots, defender_bots)
            if not available:
                deal_idx += 1
                continue

            declarer = state0.get_contract().declarer
            state0, p1_time, defender_time = self.run_playing(
                state0, p1_bots, defender_bots
            )
            p1_times.append(np.array(p1_time))
            defender_times.append(np.array(defender_time))
            
            
            restart_all_bots(defender_bots)
            # Close table.
            state1, available = self.run_bidding(cur_trajectory, p2_bots, defender_bots)
            if not available:
                deal_idx += 1
                continue

            state1, p2_time, defender_time = self.run_playing(
                state1, p2_bots, defender_bots
            )
            p2_times.append(np.array(p2_time))
            defender_times.append(np.array(defender_time))
            
            deal_idx += 1
            
            logger.write(
                f"Deal {deal_idx},State0:\n{state0}\nstate1:\n{state1}\ntricks: {state0.num_declarer_tricks()} : {state1.num_declarer_tricks()}\nscores: {state0.scores()[declarer]}: {state1.scores()[declarer]}"
            )
            print(f"process {self._process_id}, {deal_idx}/{len(self.trajectories)}")

            save_items.append(DuplicateSaveItem(state0, state1))

            np.save(
                os.path.join(
                    self.save_dir, f"p1_execution_times_{self._process_id}.npy"
                ),
                np.concatenate(p1_times),
            )

            np.save(
                os.path.join(
                    self.save_dir, f"p2_execution_times_{self._process_id}.npy"
                ),
                np.concatenate(p2_times),
            )

            np.save(
                os.path.join(
                    self.save_dir,
                    f"defender_execution_times_{self._process_id}.npy",
                ),
                np.concatenate(defender_times),
            )

            with open(
                os.path.join(self.save_dir, f"items_{self._process_id}"), "wb"
            ) as fp:
                pickle.dump(save_items, fp, protocol=pickle.HIGHEST_PROTOCOL)


@hydra.main("conf", "declarer_play", version_base="1.2")
def main(args: omegaconf.OmegaConf):
    save_dir = common_utils.mkdir_with_increment(args.save_dir)
    print(save_dir)

    dataset = load_dataset(os.path.join(args.dataset_dir, args.dataset_name))

    dataset = extract_not_passed_out_trajectories(dataset)[: args.num_deals]

    trajectories_per_process = common_utils.allocate_list_uniformly(
        dataset, args.num_processes
    )

    workers:List[Worker] = []
    for i in range(args.num_processes):
        w = Worker(args, trajectories=trajectories_per_process[0], process_id=i, save_dir=save_dir)
        workers.append(w)

    for w in workers:
        w.start()

    for w in workers:
        w.join()
    
    

if __name__ == "__main__":
    main()
