"""A stub file for rela module."""

from typing import List, Optional, Dict, Tuple, overload
import torch

TensorDict = Dict[str, torch.Tensor]


class FutureReply:
    def get(self) -> TensorDict: ...

    def is_null(self) -> bool: ...


class Batcher:
    def __init__(self, batch_size: int): ...

    def send(self, t: TensorDict) -> FutureReply: ...

    def get(self) -> TensorDict: ...

    def set(self, t: TensorDict): ...


class BatchRunner:
    @overload
    def __init__(
            self,
            py_model: torch.jit.ScriptModule,
            device: str,
            max_batch_size: int,
            methods: List[str],
    ): ...

    @overload
    def __init__(self, py_model: torch.jit.ScriptModule, device: str): ...

    def add_method(self, method: str, batch_size: int): ...

    def start(self): ...

    def stop(self): ...

    def update_model(self, py_model: torch.jit.ScriptModule): ...

    def set_log_freq(self, log_freq: int): ...

    def block_call(self, method: str, t: TensorDict): ...

    def call(self, method: str, d: TensorDict) -> FutureReply: ...


class ThreadLoop: ...


class Context:
    def __init__(self): ...

    def push_thread_loop(self, env: ThreadLoop) -> int: ...

    def start(self): ...

    def pause(self): ...

    def resume(self): ...

    def join(self): ...

    def terminated(self): ...


class RNNTransition:
    obs: Dict[str, torch.Tensor]
    action: Dict[str, torch.Tensor]
    h0: Dict[str, torch.Tensor]
    reward: torch.Tensor
    terminal: torch.Tensor
    bootstrap: torch.Tensor
    seq_len: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]: ...

    def to_device(self, device: str): ...


class RNNPrioritizedReplay:
    def __init__(
            self, capacity: int, seed: int, alpha: float, beta: float, prefetch: int
    ) -> None: ...

    def clear(self): ...

    def terminate(self): ...

    def size(self) -> int: ...

    def num_add(self) -> int: ...

    def sample(
            self, batchsize: int, device: str
    ) -> Tuple[RNNTransition, torch.Tensor]: ...

    def update_priority(self, priority: torch.Tensor): ...

    def get(self, idx: int) -> RNNTransition: ...


# TensorDict utils.
def tensor_dict_stack(vec: TensorDict, stack_dim: int) -> TensorDict: ...


def tensor_dict_eq(d0: TensorDict, d1: TensorDict) -> bool: ...


def tensor_dict_index(batch: TensorDict, i: int) -> TensorDict: ...


def tensor_dict_narrow(
        batch: TensorDict, dim: int, i: int, len: int, squeeze: bool
) -> TensorDict: ...


def tensor_dict_clone(d: TensorDict) -> TensorDict: ...


def tensor_dict_zeros_like(d: TensorDict) -> TensorDict: ...


class FFTransition:
    obs: TensorDict
    action: TensorDict
    reward: torch.Tensor
    terminal: torch.Tensor
    bootstrap: torch.Tensor
    next_obs: torch.Tensor

class FFPrioritizedReplay:
    def __init__(
            self, capacity: int, seed: int, alpha: float, beta: float, prefetch: int
    ) -> None: ...

    def clear(self): ...

    def terminate(self): ...

    def size(self) -> int: ...

    def num_add(self) -> int: ...

    def sample(
            self, batchsize: int, device: str
    ) -> Tuple[FFTransition, torch.Tensor]: ...

    def update_priority(self, priority: torch.Tensor): ...

    def get(self, idx: int) -> FFTransition: ...