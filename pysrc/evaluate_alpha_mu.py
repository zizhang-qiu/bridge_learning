"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: evaluate_alpha_mu.py
@time: 2023/12/23 16:32
"""
import pprint
import time

import set_path

set_path.append_sys_path()
import torch

import argparse
from dataclasses import dataclass
import os
import multiprocessing as mp
import numpy as np
import logging
from loguru import logger

import bridge
import bridgelearn
from common_utils.value_stats import MultiStats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", "-p", type=int, default=8)
    parser.add_argument("--dd_tolerance", type=int, default=1)
    parser.add_argument("--num_deals", type=int, default=200)
    parser.add_argument("--num_worlds", "-w", type=int, default=20)
    parser.add_argument("--num_max_moves", "-m", type=int, default=2)
    parser.add_argument("--early_cut", action="store_true")
    parser.add_argument("--root_cut", action="store_true")
    parser.add_argument("--save_dir", type=str, default="evaluation/exp1")
    parser.add_argument("--contract", type=str, default="3N")

    return parser.parse_args()


@dataclass
class EvaluateConfig:
    num_deals: int
    dd_tolerance: int
    num_worlds: int
    num_max_moves: int
    early_cut: bool
    root_cut: bool
    save_dir: str
    contract_str: str


def get_contract_from_str(contract_str: str) -> bridge.Contract:
    level = int(contract_str[0])
    denomination_str = ["C", "D", "H", "S", "N"]
    denomination = bridge.Denomination(denomination_str.index(contract_str[1].upper()))
    contract = bridge.Contract()
    contract.level = level
    contract.denomination = denomination
    contract.declarer = bridge.Seat.SOUTH
    contract.double_status = bridge.DoubleStatus.UNDOUBLED
    return contract


args = parse_args()
# print(vars(args))
save_dir = args.save_dir
logger.add(os.path.join(save_dir, "log.txt"), enqueue=True)


class Worker(mp.Process):
    def __init__(self, ev_cfg: EvaluateConfig,
                 num_deals_played: mp.Value,
                 num_deals_win_by_alpha_mu: mp.Value,
                 stats: MultiStats,
                 pid: int):
        super().__init__()
        self.ev_cfg = ev_cfg
        self.stats = stats
        self.num_deals_played = num_deals_played
        self.num_deals_win_by_alpha_mu = num_deals_win_by_alpha_mu
        self.process_id = pid

    def run(self):
        # logger = Logger(os.path.join(self.ev_cfg.save_dir, "1.txt"), auto_line_feed=True)
        # stats = MultiStats()
        np.random.seed(self.process_id)
        contract = get_contract_from_str(self.ev_cfg.contract_str)

        resampler = bridgelearn.UniformResampler(1)

        pimc_cfg = bridgelearn.PIMCConfig()
        pimc_cfg.num_worlds = self.ev_cfg.num_worlds
        pimc_cfg.search_with_one_legal_move = False
        pimc_bot = bridgelearn.PIMCBot(resampler, pimc_cfg)

        alpha_mu_cfg = bridgelearn.AlphaMuConfig()
        alpha_mu_cfg.num_worlds = self.ev_cfg.num_worlds
        alpha_mu_cfg.num_max_moves = self.ev_cfg.num_max_moves
        alpha_mu_cfg.search_with_one_legal_move = False
        alpha_mu_cfg.early_cut = self.ev_cfg.early_cut
        alpha_mu_cfg.root_cut = self.ev_cfg.root_cut
        alpha_mu_bot = bridgelearn.AlphaMuBot(resampler, alpha_mu_cfg)
        while self.num_deals_played.value < self.ev_cfg.num_deals:

            state1 = self.generate_state(contract)

            state2 = state1.clone()
            random_num = np.random.randint(0, 10000)
            resampler.reset_with_params({"seed": str(random_num)})
            while not state1.is_terminal():
                if bridgelearn.is_acting_player_declarer_side(state1):
                    st = time.perf_counter()
                    move = alpha_mu_bot.act(state1)
                    ed = time.perf_counter()
                    self.stats.feed("alpha_mu_time", ed - st)
                else:
                    st = time.perf_counter()
                    move = pimc_bot.act(state1)
                    ed = time.perf_counter()
                    self.stats.feed("pimc_time", ed - st)
                # print(move)
                state1.apply_move(move)

            # print(state1)
            # self.stats.save_all(self.ev_cfg.save_dir)

            resampler.reset_with_params({"seed": str(random_num)})
            while not state2.is_terminal():
                st = time.perf_counter()
                move = pimc_bot.act(state2)
                ed = time.perf_counter()
                self.stats.feed("pimc_time", ed - st)
                state2.apply_move(move)
            # print(state2)

            is_declarer_win_state1 = state1.scores()[contract.declarer] > 0
            is_declarer_win_state2 = state2.scores()[contract.declarer] > 0

            if is_declarer_win_state1 != is_declarer_win_state2:
                with self.num_deals_played.get_lock():
                    self.num_deals_played.value += 1

                if is_declarer_win_state1:
                    with self.num_deals_win_by_alpha_mu.get_lock():
                        self.num_deals_win_by_alpha_mu.value += 1
                logger.info(
                    f"Deal No.{self.num_deals_played.value}\nstate1\n{state1}\ntrajectory:{state1.uid_history()}\n"
                    f"state2\n{state2}\ntrajectory:{state2.uid_history()}\n"
                    f"seed: {random_num}\n"
                    f"num_win_by_alpha_mu: {self.num_deals_win_by_alpha_mu.value} / {self.num_deals_played.value}")
            else:
                logger.info(f"{self.num_deals_played.value} deals have been played,"
                            f"num_win_by_alpha_mu:{self.num_deals_win_by_alpha_mu.value} / {self.num_deals_played.value}")

    def generate_state(self, contract: bridge.Contract) -> bridge.BridgeState:
        while True:
            deal = np.random.permutation(bridge.NUM_CARDS)
            state = bridgelearn.construct_state_from_deal(deal.tolist(), bridge.default_game)
            ddt = state.double_dummy_results()
            if abs(ddt[contract.denomination][contract.declarer] - (contract.level + 6)) <= self.ev_cfg.dd_tolerance:
                for uid in [52, 52, bridge.bid_index(contract.level, contract.denomination) + 52, 52, 52, 52]:
                    move = bridge.default_game.get_move(uid)
                    state.apply_move(move)
                return state


class Worker2(mp.Process):
    def __init__(self):
        super().__init__()

    def run(self):
        logger.info("123")


if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg = EvaluateConfig(args.num_deals,
                         args.dd_tolerance,
                         args.num_worlds,
                         args.num_max_moves,
                         args.early_cut,
                         args.root_cut,
                         save_dir,
                         args.contract)

    num_deals_played = mp.Value('i', 0)
    num_deals_win_by_alpha_mu = mp.Value('i', 0)
    stats = MultiStats()
    logger.info(f"Evaluate config:\n{cfg}\n")

    workers = []
    for i in range(args.num_processes):
        w = Worker(cfg, num_deals_played, num_deals_win_by_alpha_mu, stats, i)
        workers.append(w)

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    stats.save_all(save_dir)
