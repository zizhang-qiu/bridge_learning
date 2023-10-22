"""
@file:saver
@author:qzz
@date:2023/3/21
@encoding:utf-8
"""
import os
from typing import List, Tuple

import torch


class TopKSaver:
    def __init__(self, save_dir: str, topk: int):
        self.save_dir = save_dir
        self.topk = topk
        self.worse_perf = -float("inf")
        self.worse_perf_idx = 0
        self.perfs = [self.worse_perf]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save(self, state_dict, perf, save_latest=False, force_save_name=None):
        if force_save_name is not None:
            weight_name = f"{force_save_name}.pth"
            if state_dict is not None:
                torch.save(state_dict, os.path.join(self.save_dir, weight_name))

        if save_latest:
            weight_name = "latest.pth"
            if state_dict is not None:
                torch.save(state_dict, os.path.join(self.save_dir, weight_name))

        if perf <= self.worse_perf:
            # print('i am sorry')
            # [print(i) for i in self.perfs]
            return False

        weight_name = f"model{self.worse_perf_idx}.pth"
        if state_dict is not None:
            torch.save(state_dict, os.path.join(self.save_dir, weight_name))

        if len(self.perfs) < self.topk:
            self.perfs.append(perf)
            return True

        # neesd to replace
        self.perfs[self.worse_perf_idx] = perf
        worse_perf = self.perfs[0]
        worse_perf_idx = 0
        for i, perf in enumerate(self.perfs):
            if perf < worse_perf:
                worse_perf = perf
                worse_perf_idx = i

        self.worse_perf = worse_perf
        self.worse_perf_idx = worse_perf_idx
        return True
