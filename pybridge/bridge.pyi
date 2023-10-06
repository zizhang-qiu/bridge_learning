"""
A stub file for bridge library.
"""
from typing import Dict, List

Player = int
Action = int


class PlayerAction:
    player: Player
    action: Action


class Contract:
    level: int
    denomination: int
    double_status: int
    declarer: Player

    def index(self) -> int: ...


class BridgeState:
    def __init__(self, is_dealer_vulnerable: bool, is_non_dealer_vulnerable: bool): ...

    def is_terminal(self) -> bool: ...

    def current_phase(self) -> int: ...

    def current_player(self) -> Player: ...

    def contract_index(self) -> int: ...

    def current_contract(self) -> Contract: ...

    def apply_action(self, action: Action): ...

    def legal_actions(self) -> List[Action]: ...

    def score_for_contracts(self, player: Player, contracts: List[int]) -> List[int]: ...

    def history(self) -> List[Action]: ...

    def full_history(self) -> List[PlayerAction]: ...

    def double_dummy_results(self, dds_order: bool = False): ...


class BridgeGame:
    def __init__(self, params: Dict[str, str]): ...

    def num_distinct_actions(self) -> int: ...

    def max_chance_outcomes(self) -> int: ...

    def max_utility(self) -> int: ...

    def min_utility(self) -> int: ...

    def new_initial_state(self) -> BridgeState: ...


def score(contract: Contract, declarer_tricks: int, is_vulnerable: bool) -> int: ...


def bid(level: int, denomination: int) -> int: ...


def bid_level(bid_: int) -> int: ...


def bid_denomination(bid_: int) -> int: ...


def call_string(call: int) -> str: ...


def card_suit(card: int) -> int: ...


def card_rank(card: int) -> int: ...


def card_string(card: int) -> str: ...


def card_index(suit: int, rank: int) -> int: ...


def partnership(player: Player) -> int: ...


def partner(player: Player) -> int: ...


example_cards: List[List[int]]
example_ddts: List[List[int]]

NUM_PLAYERS: int
NUM_DENOMINATIONS: int
NUM_CARDS: int
NUM_BID_LEVELS: int
NUM_BIDS: int
NUM_CALLS: int
NUM_OTHER_CALLS: int
NUM_SUITS: int
NUM_DOUBLE_STATUS: int
NUM_PARTNERSHIPS: int
NUM_CARDS_PER_SUIT: int
NUM_CARDS_PER_HAND: int
NUM_CONTRACTS: int
NUM_VULNERABILITIES: int
NUM_TRICKS: int
