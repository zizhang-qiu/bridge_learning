"""
A stub file for bridge library.
"""

from enum import IntEnum
from typing import Dict, List, Tuple, Optional, overload

Player = int
Action = int

class Seat(IntEnum):
    NORTH = ...
    EAST = ...
    SOUTH = ...
    WEST = ...

class Denomination(IntEnum):
    INVALID_DENOMINATION = ...
    CLUBS_TRUMP = ...
    DIAMONDS_TRUMP = ...
    HEARTS_TRUMP = ...
    SPADES_TRUMP = ...
    NO_TRUMP = ...

class Suit(IntEnum):
    INVALID_SUIT = ...
    CLUBS_SUIT = ...
    DIAMONDS_SUIT = ...
    HEARTS_SUIT = ...
    SPADES_SUIT = ...

class OtherCalls(IntEnum):
    NOT_OTHER_CALL = ...
    PASS = ...
    DOUBLE = ...
    REDOUBLE = ...

class DoubleStatus(IntEnum):
    UNDOUBLED = ...
    DOUBLED = ...
    REDOUBLED = ...

class Contract:
    level: int
    denomination: Denomination
    double_status: DoubleStatus
    declarer: Player

    def index(self) -> int: ...

def score(contract: Contract, declarer_tricks: int, is_vulnerable: bool) -> int: ...
def bid_index(level: int, denomination: Denomination) -> int: ...
def bid_level(bid_: int) -> int: ...
def bid_denomination(bid_: int) -> int: ...
def call_string(call: int) -> str: ...
def card_suit(card: int) -> int: ...
def card_rank(card: int) -> int: ...
def card_string(card: int) -> str: ...
def card_index(suit: int, rank: int) -> int: ...
def partnership(player: Player) -> int: ...
def partner(player: Player) -> int: ...

example_deals: List[List[int]]
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
BIDDING_ACTION_BASE: int

class BridgeCard:
    def __init__(self, suit: int, rank: int): ...
    def is_valid(self) -> bool: ...
    def suit(self) -> int: ...
    def rank(self) -> int: ...
    def index(self) -> int: ...

class BridgeHand:
    def __init__(self): ...
    def cards(self) -> List[BridgeCard]: ...
    def add_card(self, card: BridgeCard): ...
    def remove_from_hand(
        self, suit: Suit, rank: int, played_cards: List[BridgeCard]
    ): ...
    def is_full_hand(self) -> bool: ...
    def high_card_points(self) -> int: ...
    def control_value(self) -> int: ...
    def zar_high_card_points(self) -> int: ...
    def is_card_in_hand(self, card: BridgeCard) -> bool: ...
    def cards_by_suits(self) -> List[List[BridgeCard]]: ...

class MoveType(IntEnum):
    INVALID = ...
    AUCTION = ...
    PLAY = ...
    DEAL = ...

class BridgeMove:
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, level: int, denomination: Denomination): ...
    @overload
    def __init__(self, move_type: MoveType, suit: Suit, rank: int): ...
    @overload
    def __init__(self, other_call: OtherCalls): ...
    def move_type(self) -> MoveType: ...
    def is_bid(self) -> bool: ...
    def bid_level(self) -> int: ...
    def bid_denomination(self) -> Denomination: ...
    def card_suit(self) -> Suit: ...
    def card_rank(self) -> int: ...
    def other_call(self) -> OtherCalls: ...

class BridgeHistoryItem:
    move: BridgeMove
    player: Player
    deal_to_player: Player
    suit: Suit
    rank: int
    level: int
    denomination: Denomination
    other_call: OtherCalls

class BridgeGame:
    def __init__(self, params: Dict[str, str]):
        """
        A bridge game.

        Acceptable parameters:

            "is_dealer_vulnerable": Whether the dealer side is vulnerable. (default false).

            "is_non_dealer_vulnerable": Whether the non-dealer side is vulnerable. (default false).

            "dealer": The dealer of the game, first player to make a call in auction phase.

            "seed": Pseudo-random number generator seed. (default -1)
        Args:
            params: A dict of parameters.
        """

    def num_distinct_actions(self) -> int: ...
    def max_chance_outcomes(self) -> int: ...
    def max_utility(self) -> int: ...
    def min_utility(self) -> int: ...
    def max_game_length(self) -> int: ...
    def min_game_length(self) -> int: ...
    def parameters(self) -> Dict[str, str]: ...
    def new_initial_state(self) -> BridgeState: ...
    def is_dealer_vulnerable(self) -> bool: ...
    def is_non_dealer_vulnerable(self) -> bool: ...
    def is_player_vulnerable(self) -> bool: ...
    def is_partnership_vulnerable(self) -> bool: ...
    def dealer(self) -> Player: ...
    def get_move(self, uid: int) -> BridgeMove: ...
    def get_chance_outcome(self, uid: int) -> BridgeMove: ...
    def pick_random_chance(
        self, chance_outcomes: Tuple[List[BridgeMove], List[float]]
    ) -> BridgeMove: ...
    def get_move_uid(self, move: BridgeMove) -> int: ...

default_game: BridgeGame

class Phase(IntEnum):
    DEAL = ...
    AUCTION = ...
    PLAY = ...
    GAME_OVER = ...

class BridgeState:
    def __init__(self, game: BridgeGame): ...
    def hands(self) -> List[BridgeHand]: ...
    def history(self) -> List[BridgeHistoryItem]: ...
    def current_player(self) -> Player: ...
    def apply_move(self, move: BridgeMove): ...
    def move_is_legal(self, move: BridgeMove) -> bool: ...
    def is_terminal(self) -> bool: ...
    def parent_game(self) -> BridgeGame: ...
    def chance_outcome_prob(self, move: BridgeMove) -> float: ...
    def apply_random_chance(self): ...
    def current_phase(self) -> Phase: ...
    def num_declarer_tricks(self) -> int: ...
    @overload
    def legal_moves(self, player: Player) -> List[BridgeMove]: ...
    @overload
    def legal_moves(self) -> List[BridgeMove]: ...
    def score_for_contracts(
        self, player: Player, contracts: List[int]
    ) -> List[int]: ...
    def double_dummy_results(self, dds_order=False) -> List[List[int]]: ...
    def uid_history(self) -> List[int]: ...
    def is_chance_node(self) -> bool: ...
    def num_cards_played(self) -> int: ...
    def clone(self) -> BridgeState: ...
    def scores(self) -> List[int]: ...
    def get_contract(self) -> Contract: ...
    def deal_history(self) -> List[BridgeHistoryItem]: ...
    def auction_history(self) -> List[BridgeHistoryItem]: ...
    def play_history(self) -> List[BridgeHistoryItem]: ...
    def is_dummy_acting(self) -> bool: ...
    def get_dummy(self) -> int: ...

class BridgeObservation:
    @overload
    def __init__(self, state: BridgeState, observing_player: Player): ...
    @overload
    def __init__(self, state: BridgeState): ...
    def cur_player_offset(self) -> int: ...
    def auction_history(self) -> List[BridgeHistoryItem]: ...
    def hands(self) -> List[BridgeHand]: ...
    def parent_game(self) -> BridgeGame: ...
    def legal_moves(self) -> List[BridgeMove]: ...
    def is_player_vulnerable(self) -> bool: ...
    def is_opponent_vulnerable(self) -> bool: ...

class ObservationEncoder: ...

class EncoderType(IntEnum):
    CANONICAL = ...

class CanonicalEncoder(ObservationEncoder):
    def __init__(self, game: BridgeGame): ...
    def shape(self) -> List[int]: ...
    def encode(self, obs: BridgeObservation) -> List[int]: ...
    def type(self) -> EncoderType: ...

class PBEEncoder(ObservationEncoder):
    def __init__(self, game: BridgeGame) -> None: ...
    def shape(self) -> List[int]: ...
    def encode(self, obs: BridgeObservation) -> List[int]:...
    def type(self)->EncoderType:...

class JPSEncoder(ObservationEncoder):
    def __init__(self, game: BridgeGame) -> None: ...
    def shape(self) -> List[int]: ...
    def encode(self, obs: BridgeObservation) -> List[int]: ...
    def type(self) -> EncoderType: ...

def get_imp(score1: int, score2: int) -> int: ...

ALL_SUITS: List[Suit]
ALL_DENOMINATIONS: List[Denomination]
ALL_SEATS: List[Seat]

default_game_params : Dict[str, str]
