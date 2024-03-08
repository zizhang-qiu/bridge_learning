"""
@author: qzz
@contact:q873264077@gmail.com
@file: solve_playing_problem.py
@time: 2024/03/05 19:15
"""
from typing import List
import numpy as np
import os
import hydra
from set_path import append_sys_path
from utils import extract_not_passed_out_trajectories

append_sys_path()

import torch
import bridge
import rela
import bridgeplay
import bridgelearn

from create_bridge import BotFactory

def rank_str_to_rank(rank_str: str) -> int:
    if rank_str.upper() == "A":
        return 12
    if rank_str.upper() == "K":
        return 11
    if rank_str.upper() == "Q":
        return 10
    if rank_str.upper() == "J":
        return 9
    if rank_str.upper() == "T":
        return 8
    rank = int(rank_str) - 2
    assert 0 <= rank < 8
    return rank


def play_move_from_str(move_str:str)->bridge.BridgeMove:
    if len(move_str) != 2:
        raise ValueError("The move is not a correct move.")
    
    suit = "CDHS".find(move_str[0])
    rank = rank_str_to_rank(move_str[1])
    
    return bridge.BridgeMove(bridge.MoveType.PLAY, bridge.Suit(suit), rank)


def get_available_algs(state: bridge.BridgeState)->List[str]:
    declarer = state.get_contract().declarer
    if bridge.partnership(state.current_player()) == bridge.partnership(declarer):
        return ["PIMC", "Alpha_Mu"]
    else:
        # opening lead
        if not state.play_history():
            return ["PIMC", "NNB-OL","RBB-OL"]
        else:
            return ["PIMC"]

@hydra.main("conf", "solve_playing_problem", version_base="1.2")
def main(args):
    np.random.seed(1)
    game = bridge.default_game
    state = bridge.BridgeState(game)

    dataset_dir = r"D:/Projects/bridge_research/expert"
    with open(os.path.join(dataset_dir, "test.txt"), "r") as f:
        lines = f.readlines()
    test_dataset = []

    for i in range(len(lines)):
        line = lines[i].split(" ")
        test_dataset.append([int(x) for x in line])

    test_dataset = extract_not_passed_out_trajectories(test_dataset)

    random_index = np.random.randint(0, len(test_dataset))

    random_deal = test_dataset[random_index]

    idx = 0
    while state.current_phase() != bridge.Phase.PLAY:
        uid = random_deal[idx]
        if state.is_chance_node():
            move = game.get_chance_outcome(uid)
        else:
            move = game.get_move(uid)
        state.apply_move(move)
        idx += 1

    bot_factory: BotFactory = hydra.utils.instantiate(args.bot_factory)
    
    # print(state)
    while not state.is_terminal():
        if state.is_dummy_acting():
            observing_player = state.get_dummy()
        else:
            observing_player = state.current_player()
        observation = bridge.BridgeObservation(state, observing_player)

        print("Your observation:")
        print(observation)
        print("Your legal moves:")
        legal_moves = observation.legal_moves()
        print(legal_moves)
        print("Available algorithms are:")
        available_algs = get_available_algs(state)
        print(available_algs)
        reply = input("Choose an algorithm to get move or input a move:\n")

        if reply in available_algs:
            # Human choose an algorithm
            
            bot = bot_factory.create_bot(reply, player_id=state.current_player())
            bot.restart()
            move = bot.step(state)
            print(f"Algorithms choose move: {move}")
        else:
            # Human choose a move
            move = play_move_from_str(reply)
        state.apply_move(move)


if __name__ == "__main__":
    main()
