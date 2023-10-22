#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:array_utils.py
@time:2023/02/17
"""
from math import sqrt
from typing import Tuple, Union, List

import numpy as np


def multiple_shuffle(*args: Union[List, np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Shuffle multiple arrays in same order

    Args:
        *args (Vector): array to be shuffled

    Returns:

    """
    # check if their lengths are same
    lengths = [len(arg) for arg in args]
    assert len(set(lengths)) == 1
    indices = np.random.permutation(lengths[0])
    ret = []
    for arg in args:
        if not isinstance(arg, np.ndarray):
            arg = np.array(arg)
        ret.append(arg[indices])
    return tuple(ret)


def get_avg_and_sem(arr: Union[List, np.ndarray]) -> Tuple[float, float]:
    """
    Get the average and standard error of the mean of a list or array.
    Args:
        arr: The array contains data.

    Returns:
        The average and sem.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    assert arr.ndim == 1
    avg = np.mean(arr)
    sem = np.std(arr, ddof=1) / sqrt(arr.size)
    return avg.item(), sem
