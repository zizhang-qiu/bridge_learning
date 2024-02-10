"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: bba_link.py
@time: 2024/2/4 11:02
"""
import os
from enum import IntEnum

import clr

# Define constants for system types and conventions
C_NS = 0
C_WE = 1
C_INTERPRETED = 13

F_MIN_HCP = 102  # the minimum amount of HCP of the indicated player;

F_MAX_HCP = 103  # maximum amount of HCP of the indicated player;

F_MIN_PKT = 104  # the minimum number of balance points of the indicated player;

F_ZGLOSZONE_ASY = 106  # number of reported aces;

F_ZGLOSZONE_KROLE = 107  # number of reported kings;

F_ZGLOSZONA_KROTKOSC = 108  # reported suit of shortness;


class SystemType(IntEnum):
    T_21GF = 0
    T_SAYC = 1
    T_WJ = 2
    T_PC = 3


class Scoring(IntEnum):
    SCORING_MATCH_POINTS = 0
    SCORING_IMP = 1


class Vulnerability(IntEnum):
    NONE = 0
    WE = 1
    NS = 2
    BOTH = 3


class Denomination(IntEnum):
    C_CLUBS = 0
    C_DIAMONDS = 1
    C_HEARTS = 2
    C_SPADES = 3
    C_NT = 4


DENOMINATION_MAP = {"c": Denomination.C_CLUBS,
                    "d": Denomination.C_DIAMONDS,
                    "h": Denomination.C_HEARTS,
                    "s": Denomination.C_SPADES,
                    "n": Denomination.C_NT,
                    "nt": Denomination.C_NT}
DENOMINATION_STR = "CDHSN"
NUM_DENOMINATIONS = 5


class OtherCall(IntEnum):
    C_PASS = 0
    C_DOUBLE = 1
    C_REDOUBLE = 2


def get_bid(call_str: str) -> int:
    if call_str.lower() in ["p", "pass"]:
        return OtherCall.C_PASS
    if call_str.lower() in ["d", "double", "dbl", "x"]:
        return OtherCall.C_DOUBLE
    if call_str.lower() in ["r", "redouble", "rdbl", "xx"]:
        return OtherCall.C_REDOUBLE
    assert len(call_str) in [2, 3]
    assert call_str[0].isdigit()
    level = int(call_str[0])
    denomination = DENOMINATION_MAP[call_str[1:].lower()]
    return level * NUM_DENOMINATIONS + denomination


def get_call_string(bid: int) -> str:
    if bid == OtherCall.C_PASS:
        return "Pass"
    if bid == OtherCall.C_DOUBLE:
        return "Dbl"
    if bid == OtherCall.C_REDOUBLE:
        return "Rdbl"
    level = bid // NUM_DENOMINATIONS
    denomination = bid % 5
    return f"{level}{DENOMINATION_STR[denomination]}"


EPBot_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EPBot64")
# print(EPBot_PATH)

clr.AddReference(EPBot_PATH)
import EPBot64
from EPBot64 import EPBot

# player = EPBot()
# player.set_system_type(C_NS, SystemType.T_SAYC)
# player.set_system_type(C_WE, SystemType.T_SAYC)
# player.set_scoring(Scoring.SCORING_IMP)
# print(player.get_system_type(C_NS))
# print(player.get_system_type(C_WE))
# print(player.get_scoring())
#
# hand = ["", "AJ2", "AQ9765", "KQJ2"]
#
# dealer = 0
# vulnerability = Vulnerability.NONE

# conventions_list = load_conventions(r"D:\BiddingAnalyser\WBridge5-Sayc.bbsa")
# for convention, selected in conventions_list.items():
#     if selected:
#         player.set_conventions(C_NS, convention, True)
#         player.set_conventions(C_WE, convention, True)
#
# player.new_hand(0, hand, dealer, vulnerability)
#
# example_bidding_sequence = ["1d", "p", "1h", "p", "2c", "p", "3nt", "p"]
# for i, bid_str in enumerate(example_bidding_sequence):
#     pos = i % 4
#     player.set_bid(pos, get_bid(bid_str))
#
# new_bid = player.get_bid()
# print(new_bid)
# player.interpret_bid(new_bid)
#
# meaning = player.get_info_meaning(C_INTERPRETED)
# print(meaning)
#
# feature = list(player.get_info_feature(0))
# print(feature)
# print(len(feature))
# min_hcp = feature[102]
# max_hcp = feature[103]
# print(min_hcp, max_hcp)
#
# for pos in range(4):
#     print(f"pos = {pos}:")
#
#     feature = list(player.get_info_feature(pos))
#     min_hcp = feature[102]
#     max_hcp = feature[103]
#     print(f"min hcp={min_hcp}, max_hcp={max_hcp}")
#
#     honors = list(player.get_info_honors(pos))
#     print("honors:", honors, sep="\n")
#
#     min_length = list(player.get_info_min_length(pos))
#     print("min_length:", min_length, sep="\n")
#
#     max_length = list(player.get_info_max_length(pos))
#     print("max_length:", max_length, sep="\n")
#
#     probable_length = list(player.get_info_probable_length(pos))
#     print("probable_length:", probable_length, sep="\n")
#
#     suit_power = list(player.get_info_suit_power(pos))
#     print("suit_power:", suit_power, sep="\n")
#
# print(get_call_string(new_bid))
