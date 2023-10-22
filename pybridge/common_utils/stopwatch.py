from collections import defaultdict
from datetime import datetime

import numpy as np


def millis_interval(start: datetime, end: datetime) -> float:
    """
    Get milliseconds interval of 2 datetime instance
    Args:
        start: The start datetime instance
        end: The end datetime instance

    Returns:
        float: The interval in milliseconds unit.
    """
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


class Stopwatch:
    def __init__(self):
        """Implementation of stopwatch."""
        self.last_time = datetime.now()
        self.times = defaultdict(list)
        self.keys = []

    def reset(self):
        """Reset the stopwatch, clear all keys"""
        self.last_time = datetime.now()
        self.times = defaultdict(list)
        self.keys = []

    def time(self, key: str):
        """
        Get the internal for the key event.
        Args:
            key: The name of the event.
        """
        if key not in self.times:
            self.keys.append(key)
        self.times[key].append(millis_interval(self.last_time, datetime.now()))
        self.last_time = datetime.now()

    def summary(self):
        """Print summary and reset the stopwatch."""
        num_elems = -1
        total = 0
        max_key_len = 0
        for k, v in self.times.items():
            if num_elems == -1:
                num_elems = len(v)

            assert len(v) == num_elems
            total += np.sum(v)
            max_key_len = max(max_key_len, len(k))

        print("@@@Time")
        for k in self.keys:
            v = self.times[k]
            print(
                f"{k.ljust(max_key_len)}: {int(np.mean(v))} ms, {100.0 * np.sum(v) / total:.2f}%."
            )
        print(f"@@@total time per iter: {float(total) / num_elems:.2f} ms")
        self.reset()
