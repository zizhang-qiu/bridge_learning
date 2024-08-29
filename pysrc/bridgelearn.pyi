from typing import Dict, List, Optional, Union, overload

import torch
import pyrela

import bridge

GameParameters = Dict[str, str]
TensorDict = Dict[str, torch.Tensor]


class BridgeData:
    deal: List[int]
    ddt: List[int]


class BridgeDataset:
    def __init__(self, deals: List[List[int]], ddts: Optional[List[List[int]]]): ...

    def next(self) -> BridgeData: ...

    def size(self) -> int: ...


class EnvSpec:
    num_players: int
    num_partnerships: int


class GameEnv:
    pass


class BridgeEnvOptions:
    bidding_phase: bool
    playing_phase: bool
    encoder: str
    verbose: bool
    max_len: int


class BridgeEnv(GameEnv):
    def __init__(self, params: GameParameters, options: BridgeEnvOptions): ...

    def parameters(self) -> GameParameters: ...

    def feature_size(self) -> int: ...

    def set_bridge_dataset(self, bridge_dataset: BridgeDataset): ...

    def reset_with_bridge_data(self): ...

    def reset(self): ...

    def reset_with_deck(self, deal: List[int]): ...

    def reset_with_deck_and_double_dummy_results(
            self, deal: List[int], double_dummy_results: List[int]
    ): ...

    def step(self, move: Union[bridge.BridgeMove, int]): ...

    def terminated(self) -> bool: ...

    def returns(self) -> List[int]: ...

    def current_player(self) -> int: ...

    def ble_state(self) -> bridge.BridgeState: ...

    def ble_game(self) -> bridge.BridgeGame: ...

    def ble_observation(self) -> bridge.BridgeObservation: ...

    def get_move(self, uid: int) -> bridge.BridgeMove: ...

    def last_active_player(self) -> int: ...

    def last_move(self) -> bridge.BridgeMove: ...

    def feature(self, player: int) -> TensorDict: ...

    def spec(self) -> EnvSpec: ...

    def max_num_action(self) -> int: ...


class BridgeVecEnv:
    def __init__(self): ...

    def append(self, env: BridgeEnv): ...

    def reset(self): ...

    def all_terminated(self) -> bool: ...

    def any_terminated(self) -> bool: ...

    def step(self, reply: TensorDict): ...

    def feature(self) -> TensorDict: ...

    def size(self) -> int: ...

    def display(self, num_envs: int): ...


class DuplicateEnv(GameEnv):
    @overload
    def __init__(self, params: GameParameters, options: BridgeEnvOptions) -> None: ...

    @overload
    def __init__(
            self, params: GameParameters, options: BridgeEnvOptions, dataset: BridgeDataset
    ): ...

    def set_bridge_dataset(self, bridge_dataset: BridgeDataset): ...

    def max_num_action(self) -> int: ...

    def reset(self): ...

    def step(self, action: int) -> int: ...

    def terminated(self) -> bool: ...

    def current_player(self) -> int: ...

    def player_reward(self, player: int) -> float: ...

    def rewards(self) -> List[float]: ...

    def game_index(self) -> int: ...

    def current_partnership(self) -> int: ...

    def legal_actions(self) -> List[int]: ...

    def feature(self, player: int) -> TensorDict: ...

    def spec(self) -> EnvSpec: ...


class SuperviseDataGenerator:
    def __init__(
            self,
            trajectories: List[List[int]],
            batch_size: int,
            game: bridge.BridgeGame,
            seed: int,
    ): ...

    def next_batch(self, device: str) -> TensorDict: ...

    def all_data(self, device: str) -> TensorDict: ...


class BeliefDataGen:
    def __init__(
            self, trajectories: List[List[int]], batch_size: int, game: bridge.BridgeGame
    ): ...

    def next_batch(self, device: str) -> TensorDict: ...

    def all_data(self, device: str) -> TensorDict: ...


class Actor:
    def observe_before_act(self, env: GameEnv): ...

    def act(self, env: GameEnv, current_player: int): ...

    def observe_after_act(self, env: GameEnv): ...


class BridgeA2CActor(Actor):
    def __init__(self, runner: pyrela.BatchRunner, player_idx: int) -> None: ...

    def observe_before_act(self, env: GameEnv): ...

    def act(self, env: GameEnv, current_player: int): ...

    def observe_after_act(self, env: GameEnv): ...


class AllPassActor(Actor):
    def __init__(self, player_idx: int) -> None: ...

    def observe_before_act(self, env: GameEnv): ...

    def act(self, env: GameEnv, current_player: int): ...

    def observe_after_act(self, env: GameEnv): ...


class BaselineActor(Actor):
    def __init__(self, runner: pyrela.BatchRunner, player_idx: int) -> None: ...

    def observe_before_act(self, env: GameEnv): ...

    def act(self, env: GameEnv, current_player: int): ...

    def observe_after_act(self, env: GameEnv): ...


class RandomActor(Actor):
    def __init__(self, player_idx: int): ...


class BridgeLSTMActor(Actor):
    @overload
    def __init__(self, runner: pyrela.BatchRunner, player_idx: int): ...

    @overload
    def __init__(
            self,
            runner: pyrela.BatchRunner,
            max_len: int,
            gamma: float,
            replay_buffer: pyrela.RNNPrioritizedReplay,
            player_idx: int,
    ): ...

    def reset(self, env: GameEnv): ...

    def observe_before_act(self, env: GameEnv): ...

    def act(self, env: GameEnv, current_player: int): ...


class EnvActorOptions:
    eval: bool


class EnvActor: ...


class BridgeEnvActor(EnvActor):
    def __init__(
            self, env: GameEnv, options: EnvActorOptions, actors: List[Actor]
    ) -> None: ...

    def observe_before_act(self): ...

    def act(self): ...

    def observe_after_act(self): ...

    def send_experience(self): ...

    def post_send_experience(self): ...

    def get_env(self) -> GameEnv: ...

    def history_rewards(self) -> List[List[float]]: ...

    def terminal_count(self) -> int: ...

    def history_info(self) -> List[str]: ...


class EnvActorThreadLoop(pyrela.ThreadLoop):
    def __init__(
            self,
            env_actors: List[EnvActor],
            num_game_per_env: int = -1,
            thread_idx: int = -1,
            verbose: bool = False,
    ) -> None: ...

    def main_loop(self): ...


class CloneDataGenerator:
    def __init__(
            self, replay_buffer: pyrela.RNNPrioritizedReplay, max_len: int, num_threads: int, reward_type: str
    ) -> None: ...

    def set_game_params(self, params: Dict[str, str]): ...

    def set_env_options(self, env_options: BridgeEnvOptions): ...

    def add_game(self, game_trajectory: List[int]): ...

    def start_data_generation(self, inf_loop: bool, seed: int): ...

    def terminate(self): ...

    def generate_eval_data(
            self, batch_size: int, device: str, game_trajectories: List[List[int]]
    ) -> List[pyrela.RNNTransition]: ...


def registered_encoders() -> List[str]: ...


@overload
def load_encoder(name: str, game: bridge.BridgeGame, encoder_params: Dict[str, str]) -> bridge.ObservationEncoder: ...


@overload
def load_encoder(name: str, game: bridge.BridgeGame) -> bridge.ObservationEncoder: ...


def is_encoder_registered(name: str) -> bool: ...


class FFCloneDataGenerator:
    def __init__(
            self, replay_buffer: pyrela.FFPrioritizedReplay,
            num_threads: int,
            env_options: BridgeEnvOptions,
            reward_type: str,
            gamma: float
    ) -> None: ...

    def set_game_params(self, params: Dict[str, str]): ...

    def set_env_options(self, env_options: BridgeEnvOptions): ...

    def add_game(self, game_trajectory: List[int]): ...

    def start_data_generation(self, inf_loop: bool, seed: int): ...

    def terminate(self): ...

    def generate_eval_data(
            self, batch_size: int, device: str, game_trajectories: List[List[int]]
    ) -> List[pyrela.FFTransition]: ...
