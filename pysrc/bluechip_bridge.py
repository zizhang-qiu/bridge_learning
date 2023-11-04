import re

from typing import List, Dict

from wbridge5_client import Controller

# Example session:
#
# Recv: Connecting "WBridge5" as ANYPL using protocol version 18
# Send: WEST ("WBridge5") seated
# Recv: WEST ready for teams
# Send: Teams: N/S "silent" E/W "bidders"
# Recv: WEST ready to start
# Send: Start of board
# Recv: WEST ready for deal
# Send: Board number 8. Dealer WEST. Neither vulnerable.
# Recv: WEST ready for cards
# Send: WEST's cards: S A T 9 5. H K 6 5. D Q J 8 7 6. C 7.
# Recv: WEST PASSES
# Recv: WEST ready for  NORTH's bid
# Send: EAST PASSES
# Recv: WEST ready for EAST's bid
# Send: EAST bids 1C
# Recv: WEST ready for  SOUTH's bid

# The game we support
GAME_STR = "bridge(use_double_dummy_result=False)"

# Template regular expressions for messages we receive
_CONNECT = 'Connecting "(?P<client_name>.*)" as ANYPL using protocol version 18'
_PLAYER_ACTION = ("(?P<seat>NORTH|SOUTH|EAST|WEST) "
                  "((?P<pass>PASSES)|(?P<dbl>DOUBLES)|(?P<rdbl>REDOUBLES)|bids "
                  "(?P<bid>[^ ]*)|(plays (?P<play>[23456789tjqka][cdhs])))"
                  "(?P<alert> Alert.)?")
_READY_FOR_OTHER = ("{seat} ready for "
                    "(((?P<other>[^']*)'s ((bid)|(card to trick \\d+)))"
                    "|(?P<dummy>dummy))")

# Templates for fixed messages we receive
_READY_FOR_TEAMS = "{seat} ready for teams"
_READY_TO_START = "{seat} ready to start"
_READY_FOR_DEAL = "{seat} ready for deal"
_READY_FOR_CARDS = "{seat} ready for cards"
_READY_FOR_BID = "{seat} ready for {other}'s bid"

# Templates for messages we send
_SEATED = '{seat} ("{client_name}") seated'
_TEAMS = 'Teams: N/S "north-south" E/W "east-west"'
_START_BOARD = "start of board"
_DEAL = "Board number {board}. Dealer NORTH. Neither vulnerable."
_CARDS = "{seat}'s cards: {hand}"
_OTHER_PLAYER_ACTION = "{player} {action}"
_PLAYER_TO_LEAD = "{seat} to lead"
_DUMMY_CARDS = "Dummy's cards: {}"

# BlueChip bridge protocol message constants
_SEATS = ["NORTH", "EAST", "SOUTH", "WEST"]
_TRUMP_SUIT = ["C", "D", "H", "S", "NT"]
_NUMBER_TRUMP_SUITS = len(_TRUMP_SUIT)
_SUIT = _TRUMP_SUIT[:4]
_NUMBER_SUITS = len(_SUIT)
_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_LSUIT = [x.lower() for x in _SUIT]
_LRANKS = [x.lower() for x in _RANKS]

# OpenSpiel action ids
_ACTION_PASS = 52
_ACTION_DBL = 53
_ACTION_RDBL = 54
_ACTION_BID = 55  # First bid, i.e. 1C


def _bid_to_action(action_str: str) -> int:
    """Returns an OpenSpiel action id (an integer) from a BlueChip bid string."""
    level = int(action_str[0])
    trumps = _TRUMP_SUIT.index(action_str[1:])
    return _ACTION_BID + (level - 1) * _NUMBER_TRUMP_SUITS + trumps


def _play_to_action(action_str: str) -> int:
    """Returns an OpenSpiel action id (an integer) from a BlueChip card string."""
    rank = _LRANKS.index(action_str[0])
    suit = _LSUIT.index(action_str[1])
    return rank * _NUMBER_SUITS + suit


def _action_to_string(action: int) -> str:
    """Converts OpenSpiel action id (an integer) to a BlueChip action string.

    Args:
      action: an integer action id corresponding to a bid.

    Returns:
      A string in BlueChip format, e.g. 'PASSES' or 'bids 1H', or 'plays ck'.
    """
    if action == _ACTION_PASS:
        return "PASSES"
    elif action == _ACTION_DBL:
        return "DOUBLES"
    elif action == _ACTION_RDBL:
        return "REDOUBLES"
    elif action >= _ACTION_BID:
        level = str((action - _ACTION_BID) // _NUMBER_TRUMP_SUITS + 1)
        trumps = _TRUMP_SUIT[(action - _ACTION_BID) % _NUMBER_TRUMP_SUITS]
        return "bids " + level + trumps
    else:
        rank = action // _NUMBER_SUITS
        suit = action % _NUMBER_SUITS
        return "plays " + _LRANKS[rank] + _LSUIT[suit]


def _expect_regex(controller: Controller, regex: str) -> Dict[str, str]:
    """Reads a line from the controller, parses it using the regular expression."""
    line = controller.read_line()
    match = re.match(regex, line)
    if not match:
        raise ValueError("Received '{}' which does not match regex '{}'".format(
            line, regex))
    return match.groupdict()


def _expect(controller: Controller, expected: str):
    """Reads a line from the controller, checks it matches expected line exactly."""
    line = controller.read_line()
    if expected != line:
        raise ValueError("Received '{}' but expected '{}'".format(line, expected))


def _hand_string(cards: List[int]) -> str:
    """Returns the hand of the to-play player in the state in BlueChip format."""
    if len(cards) != 13:
        raise ValueError("Must have 13 cards")
    suits = [[] for _ in range(4)]
    for card in reversed(sorted(cards)):
        suit = card % 4
        rank = card // 4
        suits[suit].append(_RANKS[rank])
    for i in range(4):
        if suits[i]:
            suits[i] = _TRUMP_SUIT[i] + " " + " ".join(suits[i]) + "."
        else:
            suits[i] = _TRUMP_SUIT[i] + " -."
    return " ".join(suits)


def _connect(controller: Controller, seat: str):
    """Performs the initial handshake with a BlueChip bot."""
    client_name = _expect_regex(controller, _CONNECT)["client_name"]
    controller.send_line(_SEATED.format(seat=seat, client_name=client_name))
    _expect(controller, _READY_FOR_TEAMS.format(seat=seat))
    controller.send_line(_TEAMS)
    _expect(controller, _READY_TO_START.format(seat=seat))


def _new_deal(controller: Controller, seat: str, hand: str, board: str):
    """Informs a BlueChip bots that there is a new deal."""
    controller.send_line(_START_BOARD)
    _expect(controller, _READY_FOR_DEAL.format(seat=seat))
    controller.send_line(_DEAL.format(board=board))
    _expect(controller, _READY_FOR_CARDS.format(seat=seat))
    controller.send_line(_CARDS.format(seat=seat, hand=hand))
