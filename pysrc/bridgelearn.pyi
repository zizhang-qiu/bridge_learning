from typing import Dict, List, Optional, Union

import torch

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

class BridgeEnv:
    def __init__(self, params: GameParameters, verbose: bool): ...
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
    def feature(self) -> TensorDict: ...

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
