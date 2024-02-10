"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: rule_based_bot.py
@time: 2024/2/6 9:50
"""
import os
from typing import List, Dict, NamedTuple

import numpy as np

import set_path

set_path.append_sys_path()

import bridge
import torch
import rela
import bridgelearn
import bridgeplay

from rule_based_resampler import RuleBasedResampler, get_no_play_trajectory
from utils import load_dataset
import bba_bot


class RuleBasedBot(bridgeplay.PlayBot):
    def __init__(self, game: bridge.BridgeGame,
                 bidding_system: List[int],
                 conventions: Dict[str, int],
                 evaluator: bridgeplay.DDSEvaluator,
                 cfg: bridgeplay.BeliefBasedOpeningLeadBotConfig):
        super().__init__()
        self._game = game
        self._bidding_system = bidding_system
        self._conventions = conventions
        self._resampler = RuleBasedResampler(self._game, self._bidding_system, self._conventions)
        self._uniform_resampler = bridgeplay.UniformResampler(42)
        self._evaluator = evaluator
        self._cfg = cfg

    def step(self, state: bridge.BridgeState) -> bridge.BridgeMove:
        self._resampler.resample(state)
        num_sampled_worlds, num_sample_times = 0, 0
        states = []
        while num_sample_times < self._cfg.num_max_sample \
                and num_sampled_worlds < self._cfg.num_worlds:
            resample_result = self._resampler.resample(state)
            num_sample_times += 1
            if resample_result.success:
                sampled_state = bridgeplay.construct_state_from_deal_and_original_state(resample_result.result,
                                                                                        self._game, state)
                states.append(sampled_state)
                num_sampled_worlds += 1

        if self._cfg.verbose:
            print(f"After rule based sampling, {len(states)} deals are sampled.")

        # Fill remained deals with uniform sampling.
        if self._cfg.fill_with_uniform_sample:
            while num_sampled_worlds < self._cfg.num_worlds:
                resample_result = self._uniform_resampler.resample(state)
                num_sample_times += 1
                if resample_result.success:
                    sampled_state = bridgeplay.construct_state_from_deal_and_original_state(resample_result.result,
                                                                                            self._game, state)
                    states.append(sampled_state)
                    num_sampled_worlds += 1
        if self._cfg.verbose:
            print(f"After uniform sampling, {len(states)} deals are sampled.")
        assert len(states) == self._cfg.num_worlds

        legal_moves = state.legal_moves()
        # print(legal_moves)
        num_legal_moves = len(legal_moves)

        scores = [0 for _ in range(num_legal_moves)]

        # print(state.current_player())
        # print(state.get_contract())
        for sampled_state in states:
            # print(sampled_state)
            for i, move in enumerate(legal_moves):
                score = self._evaluator.rollout(sampled_state, move, state.current_player(),
                                                bridgeplay.RolloutResult.NUM_FUTURE_TRICKS)
                # print(score)
                scores[i] = scores[i] + score

        # print(scores, sep="\n")

        return legal_moves[np.argmax(scores)]


if __name__ == '__main__':
    dataset_dir = r"D:\Projects\bridge_research\expert"
    test_dataset = load_dataset(os.path.join(dataset_dir, "test.txt"))
    conventions_list = bba_bot.load_conventions("conf/bidding_system/WBridge5-SAYC.bbsa")
    bidding_system = [1, 1]

    evaluator = bridgeplay.DDSEvaluator()
    cfg = bridgeplay.BeliefBasedOpeningLeadBotConfig()
    cfg.num_worlds = 20
    cfg.num_max_sample = 100
    cfg.rollout_result = bridgeplay.RolloutResult.NUM_FUTURE_TRICKS
    cfg.fill_with_uniform_sample = True
    cfg.verbose = True
    bot = RuleBasedBot(bridge.default_game, bidding_system,
                       conventions_list, evaluator, cfg)

    s = bridgeplay.construct_state_from_trajectory(get_no_play_trajectory(test_dataset[2]), bridge.default_game)
    print(s)

    bot_move = bot.step(s)
    print(bot_move)

    dds_moves = evaluator.dds_moves(s)
    print(dds_moves)
