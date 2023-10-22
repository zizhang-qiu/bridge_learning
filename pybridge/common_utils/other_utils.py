#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:other_utils.py
@time:2023/02/19
"""
import datetime
import os
import platform
import random
from typing import List, Any

import numpy as np
import torch


def set_random_seeds(seed: int = 42):
    """Set random seeds for PyTorch and NumPy."""
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Python built-in
    random.seed(seed)


def gcd(a: int, b: int) -> int:
    """Returns the greatest common divisor of two integers a and b."""
    while b:
        a, b = b, a % b
    return a


def lcm(numbers: List[int]) -> int:
    """Returns the least common multiple of a list of integers."""
    if len(numbers) == 0:
        return 0
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = (result * numbers[i]) // gcd(result, numbers[i])
    return result


def set_omp_threads(num_threads: int = 1, verbose: bool = False) -> None:
    """
    set openmp thread to num_thread

    Args:
        num_threads: (int) the number of threads, default 1
        verbose: (bool) whether to print message

    Returns:None

    """
    system_platform = platform.system().lower()
    if system_platform.startswith("w"):
        os.system(f"set OMP_NUM_THREADS={num_threads}")
    elif system_platform.startswith("l"):
        os.system(f"export OMP_NUM_THREADS={num_threads}")
    if verbose:
        print(f"set omp_num_thread={num_threads}")


def mkdir_with_time(dir_path: str, _format: str = '%Y%m%d%H%M%S') -> str:
    """
    Make directory using time strf, return the path to the directory.
    Args:
        dir_path(str):the directory to make a new directory
        _format(str): the strf format of time

    Returns:
        str: the path to the made directory
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    time_str = datetime.datetime.now().strftime(_format)
    path = os.path.join(dir_path, time_str)
    os.mkdir(path)
    return path


def mkdir_with_increment(dir_path: str, folder_name: str = "folder") -> str:
    """
    Creates a new folder with the largest number + 1 in the given directory.

    Args:
        folder_name (str): The name of created folder.
        dir_path (str): Path to the directory where the new folder should be created.

    Returns:
        str: Path to the newly created folder.
    """

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # Get all folder names in the directory
    folder_names = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]

    # Extract the numbers from the folder names and convert them to integers
    numbers = [int(name.split('_')[-1]) for name in folder_names if name.startswith(f'{folder_name}_')]

    # If no folders exist, start counting from 1
    if not numbers:
        new_number = 1
    else:
        # Get the largest number and increment it by 1
        new_number = max(numbers) + 1

    # Create a new folder with the incremented number in the name
    new_folder_name = f"{folder_name}_{new_number}"
    new_folder_path = os.path.join(dir_path, new_folder_name)
    os.mkdir(new_folder_path)

    return new_folder_path


def allocate_tasks_uniformly(num_processes: int, num_tasks: int) -> List[int]:
    """
    Divides the given number of tasks as uniformly as possible among the given number of processes.

    Args:
        num_processes (int): The number of processes.
        num_tasks (int): The total number of tasks to be executed.

    Returns:
        List[int]: A list of integers representing the number of tasks to be allocated to each process.

    Example:
        >>> allocate_tasks_uniformly(4, 13)
        [4, 3, 3, 3]
    """
    tasks_per_process = num_tasks // num_processes
    extra_tasks = num_tasks % num_processes

    task_counts = [tasks_per_process] * num_processes
    for i in range(extra_tasks):
        task_counts[i] += 1

    return task_counts


def to_str_list(l: List[Any]) -> List[str]:
    """
    Convert a list of some type to a list of str.
    Args:
        l: The list

    Returns:
        The list of str
    """
    ret = [str(item) for item in l]
    return ret


def print_current_time(fmt: str = "%H:%M:%S"):
    """Print current time."""
    current_time = datetime.datetime.now().strftime(fmt)
    print("Current time is:", current_time)
