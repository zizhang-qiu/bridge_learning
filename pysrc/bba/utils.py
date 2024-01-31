"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: utils.py
@time: 2024/2/4 14:24
"""
from typing import Dict


def load_conventions(file_path: str) -> Dict[str, int]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    result_dict = {}
    for line in lines:
        key, value = line.strip().split(" = ")
        result_dict[key] = int(value)

    return result_dict
