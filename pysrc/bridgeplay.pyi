from typing import List, Dict, Optional
import bridge
import rela


class Resampler:
    pass


class ResampleResult:
    success: bool
    result: List[int]


class UniformResampler(Resampler):
    def __init__(self, seed: int):
        ...

    def resample(self) -> ResampleResult:
        ...

    def reset_with_params(self, params: Dict[str, str]): ...


class SearchResult:
    moves: List[bridge.BridgeMove]
    scores: List[int]


class PlayBot:
    def step(self, state: bridge.BridgeState) -> bridge.BridgeMove: ...


class DDSBot(PlayBot):
    def __init__(self):
        ...

    def step(self, state: bridge.BridgeState) -> bridge.BridgeMove: ...


class PIMCConfig:
    num_worlds: int
    search_with_one_legal_move: bool


class PIMCBot(PlayBot):
    def __init__(self, resampler: Resampler, cfg: PIMCConfig):
        ...

    def search(self, state: bridge.BridgeState) -> SearchResult:
        ...

    def step(self, state: bridge.BridgeState) -> bridge.BridgeMove: ...


class OutcomeVector:
    game_status: List[int]
    possible_world: List[bool]
    move: bridge.BridgeMove

    def score(self) -> float: ...


class ParetoFront:
    def __init__(self, outcome_vectors: Optional[List[OutcomeVector]]): ...

    def size(self) -> int: ...

    def insert(self, outcome_vector: OutcomeVector) -> bool: ...

    def empty(self) -> bool: ...

    def score(self) -> float: ...

    def best_outcome(self) -> OutcomeVector: ...

    def set_move(self, move: bridge.BridgeMove): ...

    @staticmethod
    def pareto_front_with_one_outcome_vector(possible_worlds: List[bool], fill_value: int) -> ParetoFront: ...

    def serialize(self) -> str: ...

    @staticmethod
    def deserialize(sr_str: str) -> ParetoFront: ...


def pareto_front_min(lhs: ParetoFront, rhs: ParetoFront) -> ParetoFront: ...


def pareto_front_max(lhs: ParetoFront, rhs: ParetoFront) -> ParetoFront: ...


def pareto_front_dominate(lhs: ParetoFront, rhs: ParetoFront) -> bool: ...


class AlphaMuConfig:
    num_max_moves: int
    num_worlds: int
    search_with_one_legal_move: bool
    root_cut: bool
    early_cut: bool


class BridgeStateWithoutHiddenInfo:
    def __init__(self, state: bridge.BridgeState): ...

    def uid_history(self) -> List[int]: ...

    def apply_move(self, move: bridge.BridgeMove): ...

    def legal_moves(self) -> List[bridge.BridgeMove]: ...

    def current_player(self) -> bridge.Player: ...

    def is_terminal(self) -> bool: ...

    def num_declarer_tricks(self) -> int: ...

    def get_contract(self) -> bridge.Contract: ...

    def serialize(self) -> str: ...

    def deserialize(self, sr_str: str, game: bridge.BridgeGame): ...


class TranspositionTable:
    def __init__(self): ...

    def table(self) -> Dict[BridgeStateWithoutHiddenInfo, ParetoFront]: ...

    def serialize(self) -> str: ...

    def deserialize(self, sr_str: str, game: bridge.BridgeGame): ...


class AlphaMuBot(PlayBot):
    def __init__(self, resampler: Resampler, cfg: AlphaMuConfig): ...

    def step(self, state: bridge.BridgeState) -> bridge.BridgeMove: ...

    def search(self, state: bridge.BridgeState) -> ParetoFront: ...

    def set_tt(self, tt: TranspositionTable): ...

    def get_tt(self) -> TranspositionTable: ...


def construct_state_from_deal(deal: List[int], game: bridge.BridgeGame) -> bridge.BridgeState: ...


def construct_state_from_trajectory(trajectory: List[int], game: bridge.BridgeGame) -> bridge.BridgeState: ...


def is_acting_player_declarer_side(state: bridge.BridgeState) -> bool: ...


class TorchActor:
    def __init__(self, runner: rela.BatchRunner): ...

    def get_policy(self, obs: rela.TensorDict) -> rela.TensorDict: ...

    def get_belief(self, obs: rela.TensorDict) -> rela.TensorDict: ...


class TorchActorResampler:
    def __init__(self, torch_actor: TorchActor, game: bridge.BridgeGame, seed: int): ...

    def resample(self, state: bridge.BridgeState) -> ResampleResult: ...


class TorchOpeningLeadBotConfig:
    num_max_sample: int = 1000
    num_worlds: int = 20
    fill_with_uniform_sample: bool = True
    verbose: bool


class TorchOpeningLeadBot(PlayBot):
    def __init__(self, torch_actor: TorchActor, game: bridge.BridgeGame, seed: int, evaluator: DDSEvaluator,
                 cfg: TorchOpeningLeadBotConfig): ...

    def step(self, state: bridge.BridgeState) -> bridge.BridgeMove: ...


def load_bot(name: str, game: bridge.BridgeGame, player: bridge.Player) -> PlayBot: ...


def dds_moves(state: bridge.BridgeState) -> List[bridge.BridgeMove]: ...


class DDSEvaluator:
    def __init__(self): ...

    def rollout(self, state: bridge.BridgeState, move: bridge.BridgeMove, result_for: bridge.Player,
                rollout_result: int) -> int: ...


class ThreadedQueueInt:
    def __init__(self, max_size: int): ...

    def pop(self) -> int: ...

    def empty(self) -> bool: ...

    def size(self) -> int: ...


class OpeningLeadEvaluationThreadLoop(rela.ThreadLoop):
    def __init__(self,
                 dds_evaluator: DDSEvaluator,
                 bot: PlayBot,
                 game: bridge.BridgeGame,
                 trajectories: List[List[int]],
                 queue: ThreadedQueueInt,
                 thread_idx: int = 0,
                 verbose: bool = False): ...
