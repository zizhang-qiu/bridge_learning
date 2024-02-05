"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: bba_bot.py
@time: 2024/2/4 16:23
"""
import os
from typing import List, Dict, Union

import set_path

set_path.append_sys_path()

import bridge

import bridgeplay

from bba import EPBot, OtherCall, C_NS, C_WE, load_conventions


def hand_to_epbot_hand(hand: bridge.BridgeHand) -> List[str]:
    card_str_by_suit = [[] for _ in range(bridge.NUM_SUITS)]
    for card in hand.cards():
        card_str_by_suit[card.suit()].append(repr(card)[1])
    res = []
    for card_str in card_str_by_suit:
        res.append("".join(card_str))
    return res


def epbot_bid_to_bridge_move(bid: int) -> bridge.BridgeMove:
    if bid == OtherCall.C_PASS:
        return bridge.BridgeMove(bridge.OtherCalls.PASS)
    if bid == OtherCall.C_DOUBLE:
        return bridge.BridgeMove(bridge.OtherCalls.DOUBLE)
    if bid == OtherCall.C_REDOUBLE:
        return bridge.BridgeMove(bridge.OtherCalls.REDOUBLE)

    level = bid // bridge.NUM_DENOMINATIONS
    denomination = bridge.Denomination(bid % bridge.NUM_DENOMINATIONS)
    return bridge.BridgeMove(level, denomination)


def bridge_move_to_epbot_bid(move: bridge.BridgeMove) -> int:
    assert move.move_type() == bridge.MoveType.AUCTION
    if move.other_call() == bridge.OtherCalls.PASS:
        return OtherCall.C_PASS
    if move.other_call() == bridge.OtherCalls.DOUBLE:
        return OtherCall.C_DOUBLE
    if move.other_call() == bridge.OtherCalls.REDOUBLE:
        return OtherCall.C_REDOUBLE
    level = move.bid_level()
    denomination = move.bid_denomination()
    return level * bridge.NUM_DENOMINATIONS + denomination.value


def get_opening_bid(state: bridge.BridgeState) -> Union[bridge.BridgeHistoryItem, None]:
    history = state.history()
    for item in history:
        if item.move.is_bid():
            return item
    return None


class BBABot(bridgeplay.PlayBot):
    def __init__(self, player_id: int, game: bridge.BridgeGame, bidding_system: List[int],
                 conventions: Dict[str, int]):
        super().__init__()
        self.ep_bot = EPBot()
        self.bidding_system = bidding_system
        self.conventions = conventions
        self.player_id = player_id
        self._game = game
        self._state = bridge.BridgeState(game)
        self._board = 0
        self._num_actions = bridge.NUM_CARDS

    def restart(self):
        if not self._state.history():
            return
        self.ep_bot = EPBot()
        self.ep_bot.set_system_type(C_NS, self.bidding_system[C_NS])
        self.ep_bot.set_system_type(C_WE, self.bidding_system[C_WE])
        for convention, selected in self.conventions.items():
            if selected:
                self.ep_bot.set_conventions(C_NS, convention, True)
                self.ep_bot.set_conventions(C_WE, convention, True)
        self._num_actions = bridge.NUM_CARDS
        self._state = bridge.BridgeState(self._game)

    def step(self, state: bridge.BridgeState) -> bridge.BridgeMove:
        # current_player = state.current_player()
        self.inform_state(state)
        new_bid = self.ep_bot.get_bid()
        # print(list(self.ep_bot.get_hand(0)))
        return epbot_bid_to_bridge_move(new_bid)

    def inform_state(self, state: bridge.BridgeState):
        full_history = state.uid_history()
        known_history = self._state.uid_history()
        if full_history[:len(known_history)] != known_history:
            raise ValueError(
                "Supplied state is inconsistent with bot's internal state\n"
                f"Supplied state:\n{state}\n"
                f"Internal state:\n{self._state}\n")

        for uid in full_history[len(known_history):]:
            if self._state.current_phase() == bridge.Phase.DEAL:
                move = self._game.get_chance_outcome(uid)
            else:
                move = self._game.get_move(uid)
            self._state.apply_move(move)
            if not self._state.is_chance_node():
                self._update_for_state()

    def _update_for_state(self):
        uid_history = self._state.uid_history()
        # If this is the first time we've seen the deal, send our hand.
        if len(uid_history) == 52:
            self._board += 1
            hand = state.hands()[self.player_id]
            epbot_hand = hand_to_epbot_hand(hand)
            dealer = state.parent_game().dealer()
            self.ep_bot.new_hand(self.player_id, epbot_hand, dealer, 0)

        # Send actions since last `step` call.
        history = self._state.history()
        for other_player_action in history[self._num_actions:]:
            self.ep_bot.set_bid(other_player_action.player,
                                bridge_move_to_epbot_bid(other_player_action.move),
                                True)
        self._num_actions = len(history)


if __name__ == '__main__':
    dataset_dir = r"D:\Projects\bridge_research\expert"
    with open(os.path.join(dataset_dir, "test.txt"), "r") as f:
        lines = f.readlines()
    test_dataset = []

    for i in range(len(lines)):
        line = lines[i].split(" ")
        test_dataset.append([int(x) for x in line])

    conventions_list = load_conventions(r"D:\BiddingAnalyser\WBridge5-Sayc.bbsa")  # 1225 8209 3303
    # conventions_list = load_conventions(r"D:\BiddingAnalyser\Sayc.bbsa") # 1187 8209 3284
    bots = [BBABot(i, bridge.default_game, [1, 1], conventions_list) for i in range(bridge.NUM_PLAYERS)]
    num_match = 0
    num_opening_bid_match = 0
    num_contract_match = 0
    for i, trajectory in enumerate(test_dataset):
        deal = trajectory[:bridge.NUM_CARDS]
        for bot in bots:
            bot.restart()
        state = bridgeplay.construct_state_from_deal(deal, bridge.default_game)
        # print(state)

        while state.current_phase() == bridge.Phase.AUCTION:
            move = bots[state.current_player()].step(state)
            # print(move)
            state.apply_move(move)

        opening_bid = get_opening_bid(state)
        contract = state.get_contract()
        # print(state)
        no_play_trajectory = trajectory[:-52] if len(trajectory) > bridge.default_game.min_game_length() else trajectory
        original_state = bridgeplay.construct_state_from_trajectory(no_play_trajectory, bridge.default_game)
        # print(original_state)
        original_opening_bid = get_opening_bid(original_state)
        original_contract = original_state.get_contract()
        if state.uid_history() == original_state.uid_history():
            num_match += 1
        if opening_bid is not None and original_opening_bid is not None and opening_bid == original_opening_bid:
            num_opening_bid_match += 1
        if contract == original_contract:
            num_contract_match += 1
        print(f"trajectory:{num_match}/{i + 1}, "
              f"opening_bid:{num_opening_bid_match}/{i + 1}, "
              f"contract:{num_contract_match}/{i + 1}")
