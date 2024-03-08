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
    trajectory = "16 50 6 7 33 8 49 14 45 17 26 20 21 40 42 41 5 28 18 23 32 15 51 13 19 4 1 22 31 46 10 30 39 37 35 27 11 44 38 43 9 3 25 29 47 2 24 36 48 34 0 12 58 60 62 65 66 52 67 52 52 52 12 48 4 0 11 15 51 23 35 7 39 3 47 17 24 27 45 37 1 13 16 44 10 20 49 41 5 2 8 6 36 32 25 29 33 34 40 42 30 9 26 14 19 46 28 18 22 21 43 31 50 38"
    trajectory = trajectory.split(" ")
    trajectory = [int(a) for a in trajectory]

    print(trajectory)

    state = bridgeplay.construct_state_from_trajectory(trajectory, bridge.default_game)
    print(state)
    # probs = np.random.rand(10)
    # probs /= np.sum(probs)
    # print(probs)
    # plt.bar(range(10), probs, width=1, edgecolor="black")
    # plt.axis("off")
    # plt.show()
    # colors = ["tomato", "deepskyblue"]
    # times = [2.95, 0.87, 0, 0, 4.45, 1.28]
    # plt.bar(range(len(times)), times, width=1, edgecolor="black", color=colors)
    # plt.xticks([])
    # plt.ylabel("Execution times(seconds)")
    # plt.ylim([0, 5])
    # plt.show()
