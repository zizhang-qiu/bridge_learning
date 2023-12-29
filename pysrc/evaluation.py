"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: evaluation.py
@time: 2023/12/29 16:41
"""
import pprint
import time
from typing import Dict

import set_path
from evaluate_alpha_mu import get_contract_from_str

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
    parser.add_argument("--num_processes", "-p", type=int, default=4)
    parser.add_argument("--p1", type=str, default="alpha_mu")
    parser.add_argument("--p2", type=str, default="pimc")
    parser.add_argument("--defender", "-d", type=str, default="dds")
    parser.add_argument("--num_deals", type=int, default=5)
    parser.add_argument("--num_worlds", "-w", type=int, default=20)
    parser.add_argument("--num_max_moves", "-m", type=int, default=2)
    parser.add_argument("--early_cut", action="store_true")
    parser.add_argument("--root_cut", action="store_true")
    parser.add_argument("--save_dir", type=str, default="evaluation/exp2")
    parser.add_argument("--contract", type=str, default="3N")

    return parser.parse_args()


@dataclass
class EvaluateConfig:
    num_deals: int
    num_worlds: int
    num_max_moves: int
    early_cut: bool
    root_cut: bool
    player1: str
    player2: str
    defender: str
    save_dir: str
    contract_str: str


args = parse_args()
save_dir = args.save_dir
logger.add(os.path.join(save_dir, "log.txt"), enqueue=True)


class Worker(mp.Process):
    def __init__(self, ev_cfg: EvaluateConfig,
                 num_deals_played: mp.Value,
                 num_deals_win_by_player1: mp.Value,
                 num_deals_win_by_player2: mp.Value,
                 pid: int):
        super().__init__()
        self.ev_cfg = ev_cfg
        self.num_deals_played = num_deals_played
        self.num_deals_win_by_player1 = num_deals_win_by_player1
        self.num_deals_win_by_player2 = num_deals_win_by_player2
        self.process_id = pid

    def create_player(self, player_str: str, pimc_cfg: bridgelearn.PIMCConfig, alpha_mu_cfg: bridgelearn.AlphaMuConfig,
                      resampler: bridgelearn.Resampler):
        if player_str == "alpha_mu":
            return bridgelearn.AlphaMuBot(resampler, alpha_mu_cfg)
        if player_str == "pimc":
            return bridgelearn.PIMCBot(resampler, pimc_cfg)
        if player_str == "dds":
            return bridgelearn.CheatBot()

        raise ValueError(f"Algorithm {player_str} not supported.")

    def run(self):
        # logger = Logger(os.path.join(self.ev_cfg.save_dir, "1.txt"), auto_line_feed=True)
        # stats = MultiStats()
        np.random.seed(self.process_id)
        contract = get_contract_from_str(self.ev_cfg.contract_str)

        resampler = bridgelearn.UniformResampler(1)

        pimc_cfg = bridgelearn.PIMCConfig()
        pimc_cfg.num_worlds = self.ev_cfg.num_worlds
        pimc_cfg.search_with_one_legal_move = False
        # pimc_bot = bridgelearn.PIMCBot(resampler, pimc_cfg)

        alpha_mu_cfg = bridgelearn.AlphaMuConfig()
        alpha_mu_cfg.num_worlds = self.ev_cfg.num_worlds
        alpha_mu_cfg.num_max_moves = self.ev_cfg.num_max_moves
        alpha_mu_cfg.search_with_one_legal_move = False
        alpha_mu_cfg.early_cut = self.ev_cfg.early_cut
        alpha_mu_cfg.root_cut = self.ev_cfg.root_cut
        # alpha_mu_bot = bridgelearn.AlphaMuBot(resampler, alpha_mu_cfg)

        player1 = self.create_player(self.ev_cfg.player1, pimc_cfg, alpha_mu_cfg, resampler)
        player2 = self.create_player(self.ev_cfg.player2, pimc_cfg, alpha_mu_cfg, resampler)
        defender = self.create_player(self.ev_cfg.defender, pimc_cfg, alpha_mu_cfg, resampler)
        while self.num_deals_played.value < self.ev_cfg.num_deals:

            state1 = self.generate_state(contract)

            state2 = state1.clone()
            random_num = np.random.randint(0, 10000)
            resampler.reset_with_params({"seed": str(random_num)})
            while not state1.is_terminal():
                if self.check_terminated():
                    break
                if bridgelearn.is_acting_player_declarer_side(state1):
                    st = time.perf_counter()
                    move = player1.act(state1)
                    ed = time.perf_counter()
                else:
                    st = time.perf_counter()
                    move = defender.act(state1)
                    ed = time.perf_counter()
                # print(move)
                state1.apply_move(move)

            # print(state1)
            # self.stats.save_all(self.ev_cfg.save_dir)
            if self.check_terminated():
                break
            resampler.reset_with_params({"seed": str(random_num)})
            while not state2.is_terminal():
                if self.check_terminated():
                    break
                if bridgelearn.is_acting_player_declarer_side(state2):
                    st = time.perf_counter()
                    move = player2.act(state2)
                    ed = time.perf_counter()
                else:
                    st = time.perf_counter()
                    move = defender.act(state2)
                    ed = time.perf_counter()
                state2.apply_move(move)
            # print(state2)

            if self.check_terminated():
                break

            is_declarer_win_state1 = state1.scores()[contract.declarer] > 0
            is_declarer_win_state2 = state2.scores()[contract.declarer] > 0
            with self.num_deals_played.get_lock():
                self.num_deals_played.value += 1

            if is_declarer_win_state1 != is_declarer_win_state2:

                if is_declarer_win_state1:
                    with self.num_deals_win_by_player1.get_lock():
                        self.num_deals_win_by_player1.value += 1
                else:
                    with self.num_deals_win_by_player2.get_lock():
                        self.num_deals_win_by_player2.value += 1
                logger.info(
                    f"Deal No.{self.num_deals_played.value}\nstate1\n{state1}\ntrajectory:{state1.uid_history()}\n"
                    f"state2\n{state2}\ntrajectory:{state2.uid_history()}\n"
                    f"seed: {random_num}\n"
                    f"num_win_by_player1: {self.num_deals_win_by_player1.value}\n"
                    f"num_win_by_player2: {self.num_deals_win_by_player2.value}")
            else:
                logger.info(f"{self.num_deals_played.value} deals have been played,"
                            f"num_win_by_player1: {self.num_deals_win_by_player1.value}\n"
                            f"num_win_by_player2: {self.num_deals_win_by_player2.value}")

    def generate_state(self, contract: bridge.Contract) -> bridge.BridgeState:
        while True:
            deal = np.random.permutation(bridge.NUM_CARDS)
            state = bridgelearn.construct_state_from_deal(deal.tolist(), bridge.default_game)
            ddt = state.double_dummy_results()
            if ddt[contract.denomination][contract.declarer] - (contract.level + 6) >= 0:
                for uid in [52, 52, bridge.bid_index(contract.level, contract.denomination) + 52, 52, 52, 52]:
                    move = bridge.default_game.get_move(uid)
                    state.apply_move(move)
                return state

    def check_terminated(self):
        return self.num_deals_played.value >= self.ev_cfg.num_deals


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cfg = EvaluateConfig(args.num_deals,
                         args.num_worlds,
                         args.num_max_moves,
                         args.early_cut,
                         args.root_cut,
                         args.p1,
                         args.p2,
                         args.defender,
                         save_dir,
                         args.contract)
    keys = ["pimc_time", "alpha_mu_time"]

    num_deals_played = mp.Value('i', 0)
    num_deals_win_by_player1 = mp.Value('i', 0)
    num_deals_win_by_player2 = mp.Value('i', 0)

    logger.info(f"Evaluate config:\n{cfg}\n")

    workers = []
    for i in range(args.num_processes):
        w = Worker(cfg, num_deals_played, num_deals_win_by_player1, num_deals_win_by_player2, i)
        workers.append(w)

    for w in workers:
        w.start()

    for w in workers:
        w.join()
