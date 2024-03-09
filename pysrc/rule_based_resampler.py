"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: rule_based_resampler.py
@time: 2024/2/5 9:12
"""
import os
import random
from typing import List, Dict, NamedTuple, Tuple

import numpy as np
from tqdm import trange

import bba_bot
import set_path
from bba import C_NS, C_WE
from utils import load_dataset
from redeal.redeal import *

set_path.append_sys_path()
import bridge
import torch
import rela
import bridgelearn
import bridgeplay


def get_no_play_trajectory(trajectory: List[int]) -> List[int]:
    if len(trajectory) == bridge.default_game.min_game_length():
        return trajectory
    if trajectory[-1] == bridge.OtherCalls.PASS.value + 52:
        return trajectory
    return trajectory[:-bridge.NUM_CARDS]


def bridge_hand_to_redeal_hand(bridge_hand: bridge.BridgeHand) -> str:
    cards_by_suit = bridge_hand.cards_by_suits()
    str_by_suit = [[] for _ in range(bridge.NUM_SUITS)]
    for suit in [bridge.Suit.SPADES_SUIT, bridge.Suit.HEARTS_SUIT, bridge.Suit.DIAMONDS_SUIT, bridge.Suit.CLUBS_SUIT]:
        for card in cards_by_suit[suit]:
            str_by_suit[suit].append(repr(card)[1])
    for i in range(bridge.NUM_SUITS):
        if not str_by_suit[i]:
            str_by_suit[i].append('-')
    str_by_suit.reverse()
    result = " ".join(["".join(suit_str) for suit_str in str_by_suit])
    return result


def deal_player_suit(deal_player, suit: bridge.Suit):
    if suit == bridge.Suit.CLUBS_SUIT:
        return deal_player.clubs
    if suit == bridge.Suit.DIAMONDS_SUIT:
        return deal_player.diamonds
    if suit == bridge.Suit.HEARTS_SUIT:
        return deal_player.hearts
    if suit == bridge.Suit.SPADES_SUIT:
        return deal_player.spades
    raise ValueError(f"Wrong suit: {suit}")


def get_deal_player(deal: Deal, seat: bridge.Seat):
    if seat == bridge.Seat.NORTH:
        return deal.north
    if seat == bridge.Seat.SOUTH:
        return deal.south
    if seat == bridge.Seat.EAST:
        return deal.east
    if seat == bridge.Seat.WEST:
        return deal.west
    raise ValueError(f"Wrong seat: {seat}")


def redeal_deal_to_deal(deal: Deal) -> List[int]:
    result_deal = np.zeros(bridge.NUM_CARDS, dtype=np.int32)
    for seat in bridge.ALL_SEATS:
        this_hand: Hand = get_deal_player(deal, seat)
        for i, card in enumerate(this_hand.cards()):
            result_deal[i * bridge.NUM_PLAYERS + seat.value] = bridge.card_index(bridge.Suit(3 - card.suit.value),
                                                                                 card.rank.__index__())

    return result_deal.tolist()


class InferredHandInfo(NamedTuple):
    min_hcp: int
    max_hcp: int
    honors: List[int]
    min_length: List[int]
    max_length: List[int]
    probable_length: List[int]
    suit_power: List[int]
    stoppers: List[int]
    strength: List[int]


class RuleBasedResampler(bridgeplay.Resampler):
    def __init__(self, game: bridge.BridgeGame,
                 _bidding_system: List[int], conventions: Dict[str, int]):
        super().__init__()
        self.ep_bot = bba_bot.EPBot()
        self._game = game
        self._bidding_system = _bidding_system
        self._conventions = conventions
        self.reset_with_params({})
        self._cached_history: List[int] = []
        self._cached_hand_infos: Dict[int, InferredHandInfo] = {}
        self._cached_opening_lead_hand: bridge.BridgeHand = bridge.BridgeHand()

    def reset_with_params(self, params: Dict[str, str]):
        self._cached_history: List[int] = []
        self._cached_hand_infos: Dict[int, InferredHandInfo] = {}
        self._cached_opening_lead_hand: bridge.BridgeHand = bridge.BridgeHand()
        self.ep_bot.set_system_type(C_NS, self._bidding_system[C_NS])
        self.ep_bot.set_system_type(C_WE, self._bidding_system[C_WE])
        for convention, selected in self._conventions.items():
            if selected:
                self.ep_bot.set_conventions(C_NS, convention, True)
                self.ep_bot.set_conventions(C_WE, convention, True)

    def set_bidding_system(self, _bidding_system: List[int]):
        self._bidding_system = _bidding_system

    def set_conventions(self, conventions: Dict[str, int]):
        self._conventions = conventions

    def rollout(self, trajectory: List[int]) -> Tuple[bridge.BridgeHand, Dict[int, InferredHandInfo]]:
        state = bridgeplay.construct_state_from_deal(trajectory[:bridge.NUM_CARDS], self._game)
        hands = state.hands()
        for pos, hand in enumerate(hands):
            self.ep_bot.new_hand(pos, bba_bot.hand_to_epbot_hand(hand), self._game.dealer(), 0)

        no_play_trajectory = get_no_play_trajectory(trajectory)
        for uid in no_play_trajectory[bridge.NUM_CARDS:]:
            move = self._game.get_move(uid)
            bid = bba_bot.bridge_move_to_epbot_bid(move)
            current_player = state.current_player()

            self.ep_bot.interpret_bid(bid)
            self.ep_bot.set_bid(current_player, bid)
            # meaning = self.ep_bot.get_info_meaning(current_player)
            # print("meaning: ", meaning)

            state.apply_move(move)
        # print(state)
        current_player = state.current_player()
        # print("cur player: ", current_player)
        # self.bba_bots[current_player].get_info()
        # self.get_info(current_player)
        inferred_hand_infos: Dict[int, InferredHandInfo] = {}
        for pos in range(bridge.NUM_PLAYERS):
            if pos != current_player:
                # print("pos: ", pos)
                feature = list(self.ep_bot.get_info_feature(pos))

                # print("feature:\n", feature)

                min_hcp = feature[102]
                max_hcp = feature[103]
                # print(f"min hcp={min_hcp}, max_hcp={max_hcp}")

                honors = list(self.ep_bot.get_info_honors(pos))
                # print("honors: ", honors, sep="\n")

                min_length = list(self.ep_bot.get_info_min_length(pos))
                # print("min_length:", min_length, sep="\n")

                max_length = list(self.ep_bot.get_info_max_length(pos))
                # print("max_length:", max_length, sep="\n")

                probable_length = list(self.ep_bot.get_info_probable_length(pos))
                # print("probable_length:", probable_length, sep="\n")

                suit_power = list(self.ep_bot.get_info_suit_power(pos))
                # print("suit_power:", suit_power, sep="\n")

                stoppers = list(self.ep_bot.get_info_stoppers(pos))
                # print("stoppers:", stoppers, sep="\n")

                strength = list(self.ep_bot.get_info_strength(pos))
                # print("strength:", strength, sep="\n")

                info = InferredHandInfo(min_hcp, max_hcp, honors, min_length, max_length, probable_length, suit_power,
                                        stoppers, strength)
                inferred_hand_infos[pos] = info

        opening_lead_player_hand = hands[current_player]
        self._cached_hand_infos = inferred_hand_infos
        self._cached_opening_lead_hand = opening_lead_player_hand
        self._cached_history = trajectory
        return opening_lead_player_hand, inferred_hand_infos

    def resample(self, state: bridge.BridgeState) -> bridgeplay.ResampleResult:
        assert state.num_cards_played() == 0 and state.current_phase() == bridge.Phase.PLAY
        trajectory = state.uid_history()
        if trajectory == self._cached_history:
            opening_lead_player_hand = self._cached_opening_lead_hand
            inferred_hand_infos = self._cached_hand_infos
        else:
            opening_lead_player_hand, inferred_hand_infos = self.rollout(trajectory)
        # print(inferred_hand_infos)
        predeal = {"NESW"[state.current_player()]: bridge_hand_to_redeal_hand(opening_lead_player_hand)}

        dealer = Deal.prepare(predeal)

        def accept(deal) -> bool:
            constraint: bool = True
            # player_id_to_deal_player = {
            #     0: deal.north,
            #     1: deal.east,
            #     2: deal.south,
            #     3: deal.west
            # }

            for player, hand_info in inferred_hand_infos.items():
                deal_player = get_deal_player(deal, player) # type: ignore

                hcp_constraint = hand_info.max_hcp >= deal_player.hcp >= hand_info.min_hcp
                constraint = constraint and hcp_constraint
                for suit in bridge.ALL_SUITS:
                    suit_constraint = hand_info.max_length[suit] >= len(deal_player_suit(deal_player, suit)) >= \
                                      hand_info.min_length[suit]
                    constraint = constraint and suit_constraint
            return constraint

        resample_result = bridgeplay.ResampleResult()
        try:
            deal = dealer(accept, tries=1000)
        except Exception as e:
            # print(e)
            resample_result.success = False
            resample_result.result = np.full(bridge.NUM_CARDS, -1, dtype=np.int32)
            return resample_result
        deal_array = redeal_deal_to_deal(deal)
        resample_result.result = deal_array
        resample_result.success = True
        return resample_result


if __name__ == '__main__':
    dataset_dir = r"D:\Projects\bridge_research\expert"
    test_dataset = load_dataset(os.path.join(dataset_dir, "test.txt"))

    conventions_list = bba_bot.load_conventions("conf/bidding_system/WBridge5-SAYC.bbsa")
    bidding_system = [1, 1]

    resampler = RuleBasedResampler(bridge.default_game, bidding_system, conventions_list)
    print(issubclass(RuleBasedResampler, bridgeplay.Resampler))

    # hand, infos = resampler.rollout(test_dataset[5])
    num_sample = 100
    num_total_success = 0
    for i, trajectory in enumerate(random.sample(test_dataset, 50)):
        state1 = bridgeplay.construct_state_from_trajectory(get_no_play_trajectory(trajectory), bridge.default_game)
        # print(state1)
        num_success = 0
        for j in trange(num_sample):
            resample_deal = resampler.resample(state1)
            if resample_deal.success:
                # state2 = bridgeplay.construct_state_from_trajectory(resample_deal.result, bridge.default_game)
                # print(state2)
                num_success += 1
            else:
                # print("Resample failed.")
                pass
        num_total_success += num_success
        print(f"{num_success}/{num_sample}")
    print(f"Total: {num_total_success} / {num_sample * 50}")
