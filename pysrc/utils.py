"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: utils.py
@time: 2024/2/5 15:27
"""
import os

from typing import List, Tuple, Dict, OrderedDict

import torch
import yaml


def load_dataset(file_path: str) -> List[List[int]]:
    """
    Load a dataset from a text file, e.g.

    27 46 36 43 18 22 0 20 24 2 40 41 28 16 21 10 42 32 ...

    33 6 20 31 29 38 24 25 30 50 8 48 37 51 32 44 18 41 11 ...
    Args:
        file_path(str): the path to the file.

    Returns:
        List[List[int]]: Trajectories.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    res = []
    for line in lines:
        trajectory = [int(a) for a in line.split(" ")]
        res.append(trajectory)

    return res


def load_net_conf_and_state_dict(model_dir: str, model_name: str, net_conf_filename: str = "net.yaml") \
        -> Tuple[Dict, OrderedDict]:
    with open(os.path.join(model_dir, net_conf_filename), "r") as fp:
        conf = yaml.full_load(fp)
    state_dict_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(state_dict_path)
    return conf, state_dict


def is_trajectory_not_passed_out(trajectory: List[int]):
    return trajectory[-4:] != [52 for _ in range(4)]


def extract_not_passed_out_trajectories(trajectories: List[List[int]]) -> List[List[int]]:
    res = []
    for trajectory in trajectories:
        if is_trajectory_not_passed_out(trajectory):
            res.append(trajectory)
    return res


