"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: temp2.py
@time: 2024/1/17 15:06
"""

import hydra
import numpy as np
import matplotlib.pyplot as plt
from set_path import append_sys_path

append_sys_path()

import bridge
import torch
import rela
import bridgeplay

# if __name__ == '__main__':
#     # conf = OmegaConf.load("conf/net2.yaml")
#     # print(conf)
#     #
#     # net = hydra.utils.instantiate(conf)
#     #
#     # print(net)
#     conf = OmegaConf.load("conf/game.yaml")

#     game = create_bridge.create_bridge_game(dict(conf))
#     print(game)
if __name__ == "__main__":
    a = torch.arange(12).reshape(3, 4)
    print(a)
    m = torch.argmax(a, dim=1, keepdim=True)
    print(m)
    print(m[0])