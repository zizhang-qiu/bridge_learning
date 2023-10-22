"""Utils for assertions"""
from typing import NoReturn, Iterable

import numpy as np
import torch
from torch import nn


def assert_eq(real, expected):
    """
    Assert real and expected are equal.
    Args:
        real: The real item.
        expected: The expected item.

    Returns:
        No returns.
    """
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_neq(real, expected):
    """
    Assert real and expected are not equal.
    Args:
        real: The real item.
        expected: The expected item.

    Returns:
        No returns.
    """
    assert real != expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_lt(real, expected):
    """
    Assert real is less than expected.
    Args:
        real: The real item.
        expected: The expected item.

    Returns:
        No returns.
    """
    assert real < expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_lteq(real, expected):
    """
    Assert real is less than or equal with expected.
    Args:
        real: The real item.
        expected: The expected item.

    Returns:
        No returns.
    """
    assert real <= expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_tensor_eq(t1: torch.Tensor, t2: torch.Tensor, eps=1e-6):
    """
    Assert 2 tensors are equal, with eps tolerance.
    Args:
        t1: A tensor
        t2: A tensor
        eps: The tolerance

    Returns:
        No returns.
    """
    if t1.size() != t2.size():
        print("Warning: size mismatch", t1.size(), "vs", t2.size())
        return False

    t1 = t1.cpu().numpy()
    t2 = t2.cpu().numpy()
    diff = abs(t1 - t2)
    eq = (diff < eps).all()
    if not eq:
        import pdb

        pdb.set_trace()
    assert eq


def assert_zero_grad(params):
    """
    Assert a network's params has zero grad.
    Args:
        params: network's params.

    Returns:
        No returns.
    """
    for p in params:
        if p.grad is not None:
            assert p.grad.sum().item() == 0


def assert_in(item, obj: Iterable):
    """
    Assert the item is in an iterable object.
    Args:
        item: The item.
        obj: A iterable object. Such as a list.

    Returns:
        No returns.
    """
    assert item in obj, f"item {item} not in iterable {obj}."


def assert_in_range(real, range_left, range_right) -> NoReturn:
    """
    Assert a num in a left closed right open range interval
    Args:
        real: The real number
        range_left: The left range, it is closed
        range_right: The right range, it is open

    Returns:
        No returns
    """
    assert range_left <= real < range_right, f"expected range is [{range_left}, {range_right}), the number is {real}"


def assert_network_normal(network: nn.Module):
    """
    Assert the network's parameter is normal, i.e. no inf or nan.
    Args:
        network: The network.

    Returns:

    """
    for param in network.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            assert False, "the network contains nan or inf!"
