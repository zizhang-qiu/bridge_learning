#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:value_stats.py
@time:2023/02/21
"""
import os
from typing import List, Dict, NoReturn, Union, Optional

import numpy as np
from matplotlib import pyplot as plt

ValueLike = Union[int, float]


class ValueStats:
    """implementation of stat manager"""

    def __init__(self, name: Optional[str] = None):
        """
        Initialize

        Args:
            name: the name of the stat
        """
        self._name = name
        self.counter: int = 0
        self.summation: float = 0.0
        self.stat_list: List[ValueLike] = []
        self.max_value: ValueLike = -1e38
        self.min_value: ValueLike = 1e38
        self.max_idx: Optional[int] = None
        self.min_idx: Optional[int] = None

    def feed(self, v: ValueLike):
        """
        Feed a value to stat

        Args:
            v: the value

        Returns: None

        """
        self.summation += v
        self.stat_list.append(v)
        if v > self.max_value:
            self.max_value = v
            self.max_idx = self.counter
        if v < self.min_value:
            self.min_value = v
            self.min_idx = self.counter

        self.counter += 1

    def save(self, save_dir: str, plot=False) -> NoReturn:
        """
        Save the list as npy file and plot figure if need

        Args:
            save_dir: the path to save
            plot: if True, plot figure and save as png

        """
        npy_path = os.path.join(save_dir, f"{self._name}.npy")
        np.save(npy_path, np.array(self.stat_list))
        if plot:
            plt.figure()
            x = np.arange(self.counter)
            plt.plot(x, self.stat_list)
            fig_path = os.path.join(save_dir, f"{self._name}.png")
            plt.savefig(fig_path)
            plt.close()

    def mean(self) -> float:
        """Get mean value"""
        assert self.counter != 0, "counter=0, please feed stats."
        return self.summation / self.counter

    def summary(self, info=None) -> str:
        """
        Get summary

        Args:
            info: a str appears on the start of return, can be None

        Returns:summary string, consist of name, counter, mean, minimum value and its index, maximum value and its index

        """
        info = "" if info is None else info
        name = "" if self._name is None else self._name
        if self.counter > 0:
            # try:
            return "%s%s[%4d]: avg: %8.4f, min: %8.4f[%4d], max: %8.4f[%4d]" % (
                info,
                name,
                self.counter,
                self.summation / self.counter,
                self.min_value,
                self.min_idx,
                self.max_value,
                self.max_idx,
            )
            # except BaseException:
            #     return "%s%s[Err]:" % (info, name)
        else:
            return "%s%s[0]" % (info, name)

    def reset(self):
        """Reset the values"""
        self.counter = 0
        self.summation = 0.0
        self.stat_list = []
        self.max_value = -1e38
        self.min_value = 1e38
        self.max_idx = None
        self.min_idx = None

    def __repr__(self):
        ret = f"The stat's name is {self._name}, counter={self.counter}, min value={self.min_value}, " \
              f"max_value={self.max_value}"
        return ret


class MultiStats:
    """A class manages multi stats"""

    def __init__(self):
        self.stat_names: List[str] = []
        self.stats: Dict[str, ValueStats] = {}

    def reset(self):
        self.stat_names.clear()
        self.stats.clear()

    def get(self, stat_name: str):
        assert stat_name in self.stat_names
        return self.stats[stat_name]

    def feed(self, stat_name: str, value: ValueLike):
        """
        Feed a stat with name.
        Args:
            stat_name: The name of the stat
            value: The value of the stat

        Returns:
            No returns.
        """
        if stat_name not in self.stat_names:
            value_stat = ValueStats(name=stat_name)
            self.stats[stat_name] = value_stat
            self.stat_names.append(stat_name)
        self.stats[stat_name].feed(value)

    def save_all(self, save_dir: str, plot=False):
        """
        Save all the stats.
        Args:
            save_dir: The directory to save stats.
            plot: Whether to plot the figure.

        Returns:
            No returns.
        """
        for stat_value in self.stats.values():
            stat_value.save(save_dir, plot)

    def save(self, stat_name: str, save_dir: str, plot=False):
        """
        Save a single stat
        Args:
            stat_name: The name of stat.
            save_dir: The directory to save.
            plot: Whether to plot figure.

        Returns:
            No returns.
        """
        if stat_name not in self.stat_names:
            raise ValueError(f"The stat {stat_name} has not been saved.")
        self.stats[stat_name].save(save_dir, plot)

    def __repr__(self):
        ret = f"the MultiStats consist of {len(self.stats.keys())} stats\n"
        for stat_name, stat_value in self.stats.items():
            ret += f"{stat_value}\n"
        return ret
