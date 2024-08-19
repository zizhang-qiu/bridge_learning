"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: utils.py
@time: 2024/2/5 15:27
"""

import os
import time

from typing import List, Tuple, Dict, OrderedDict

import torch
import yaml
import numpy as np
import pickle

import common_utils


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


def load_net_conf_and_state_dict(
        model_dir: str, model_name: str, net_conf_filename: str = "net.yaml"
) -> Tuple[Dict, OrderedDict]:
    with open(os.path.join(model_dir, net_conf_filename), "r") as fp:
        conf = yaml.full_load(fp)
    state_dict_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(state_dict_path)
    return conf, state_dict


def is_trajectory_not_passed_out(trajectory: List[int]) -> bool:
    """Check if a trajectory is a game which is passed out, i.e., four players make a call of pass.

    Args:
        trajectory (List[int]): The trajectory of a game.

    Returns:
        bool: Whether the trajectory is passed out.
    """
    return trajectory[-4:] != [52 for _ in range(4)]


def extract_not_passed_out_trajectories(
        trajectories: List[List[int]],
) -> List[List[int]]:
    """Extract trajectories which are not passed out from a list of trajectories.

    Args:
        trajectories (List[List[int]]): The trajectories.

    Returns:
        List[List[int]]: The extracted trajectories.
    """
    res = []
    for trajectory in trajectories:
        if is_trajectory_not_passed_out(trajectory):
            res.append(trajectory)
    return res


def tensor_dict_to_device(
        d: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    """Move a TensorDict to device.

    Args:
        d (Dict[str, torch.Tensor]): The TensorDict to be moved.
        device (str): The device to move.

    Returns:
        Dict[str, torch.Tensor]: The moved dict.
    """
    res = {}
    for k, v in d.items():
        res[k] = v.to(device)

    return res


def tensor_dict_unsqueeze(d: Dict[str, torch.Tensor], dim=0) -> Dict[str, torch.Tensor]:
    """Do torch.unsqueeze() for all tensors in a tensor dict.

    Args:
        d (Dict[str, torch.Tensor]): The tensor dict.
        dim (int, optional): The dimension for unsqueeze. Defaults to 0.

    Returns:
        Dict[str, torch.Tensor]: The unsqueezed tensor_dict.
    """
    res = {}
    for k, v in d.items():
        res[k] = torch.unsqueeze(v, dim)
    return res


def load_rl_dataset(
        usage: str, dataset_dir: str = "D:/Projects/bridge_research/dataset/rl_data"
) -> Dict[str, np.ndarray]:
    """
    Load dataset.
    Args:
        usage (str): should be one of "train", "valid", "vs_wb5_fb" or "vs_wb5_open_spiel"
        dataset_dir (str): the dir to dataset, the file names should be usage + _trajectories.npy / _ddts.npy

    Returns:
        RLDataset: The cards, ddts and par scores, combined as a dict
    """
    dataset_name = usage + ".pkl"
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"No such path: {dataset_path}, please check path or name.")

    with open(dataset_path, "rb") as fp:
        dataset: Dict[str, np.ndarray] = pickle.load(fp)

    return dataset


class Tachometer:
    def __init__(self):
        self.num_buffer = 0
        self.num_train = 0
        self.t = None
        self.total_time = 0

    def start(self):
        self.t = time.time()

    def lap(self, replay_buffer, num_train, factor):
        t = time.time() - self.t
        self.total_time += t
        num_buffer = replay_buffer.num_add()
        buffer_rate = factor * (num_buffer - self.num_buffer) / t
        train_rate = factor * num_train / t
        print(
            "Speed: train: %.1f, buffer_add: %.1f, buffer_size: %d"
            % (train_rate, buffer_rate, replay_buffer.size())
        )
        self.num_buffer = num_buffer
        self.num_train += num_train
        print(
            "Total Time: %s, %ds"
            % (common_utils.sec2str(self.total_time), self.total_time)
        )
        print(
            "Total Sample: train: %s, buffer: %s"
            % (
                common_utils.num2str(self.num_train),
                common_utils.num2str(self.num_buffer),
            )
        )
