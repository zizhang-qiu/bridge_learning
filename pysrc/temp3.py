"""
@author: qzz
@contact:q873264077@gmail.com
@file: temp3.py
@time: 2024/02/22 16:43
"""

import os
import re
from typing import List
import hydra
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import common_utils
from evaluate_declarer_play_against_wbridge5 import DuplicateSaveItem
import bridge
import bridgeplay

# @hydra.main(config_path="conf/optimizer", config_name="adam", version_base="1.2")
# def main(args):
#         opt = hydra.utils.instantiate(args)
#         print(opt)

if __name__ == "__main__":
    file_dir = "../declarer_eval/exp16"
    execution_times_arr = np.load(os.path.join(file_dir, "execution_times.npy"))
    print(len(execution_times_arr))
    print(common_utils.get_avg_and_sem(execution_times_arr))

    with open(os.path.join(file_dir, "items"), "rb") as fp:

        duplicate_items: List[DuplicateSaveItem] = pickle.load(fp)

    alpha_mu_tricks = [item.state0.num_declarer_tricks() for item in duplicate_items]
    wb_tricks = [item.state1.num_declarer_tricks() for item in duplicate_items]
    print(common_utils.get_avg_and_sem(alpha_mu_tricks))
    print(common_utils.get_avg_and_sem(wb_tricks))

    alpha_mu_scores = [
        item.state0.scores()[item.state0.get_contract().declarer]
        for item in duplicate_items
    ]
    wb_scores = [
        item.state1.scores()[item.state1.get_contract().declarer]
        for item in duplicate_items
    ]
    print(common_utils.get_avg_and_sem(alpha_mu_scores))
    print(common_utils.get_avg_and_sem(wb_scores))

    num_alpha_mu_win = 0
    num_wb_win = 0
    for item in duplicate_items:
        state0_win = item.state0.scores()[item.state0.get_contract().declarer] > 0
        state1_win = item.state1.scores()[item.state1.get_contract().declarer] > 0
        if state0_win != state1_win:
            if state0_win:
                num_alpha_mu_win += 1
            else:
                num_wb_win += 1

    print(
        f"Num difference: {num_alpha_mu_win + num_wb_win}, num_alpha_mu_win: {num_alpha_mu_win}"
    )

    #     #     main()
    #     tricks = [0, 0]
    #     scores = [0, 0]
    #     cnt = 0
    #     for i in range(4):
    #         with open(f"../declarer_eval/exp8/log_{i}.txt") as f:
    #             content = f.read()

    #         match = re.findall("tricks: (.*?) : (.*?)\n scores: (.*?):(.*?)\n", content)
    #         for m in match:
    #             tricks[0] += int(m[0])
    #             tricks[1] += int(m[1])
    #             scores[0] += int(m[2])
    #             scores[1] += int(m[3])
    #             cnt += 1

    #     print(tricks, scores, cnt)
    # arr = np.load("../declarer_eval/exp9/execution_times.npy")
    # print(arr)
    # print(common_utils.get_avg_and_sem(arr))
    # trajectory = "16 50 6 7 33 8 49 14 45 17 26 20 21 40 42 41 5 28 18 23 32 15 51 13 19 4 1 22 31 46 10 30 39 37 35 27 11 44 38 43 9 3 25 29 47 2 24 36 48 34 0 12 58 60 62 65 66 52 67 52 52 52 12 48 4 0 11 15 51 23 35 7 39 3 47 17 24 27 45 37 1 13 16 44 10 20 49 41 5 2 8 6 36 32 25 29 33 34 40 42 30 9 26 14 19 46 28 18 22 21 43 31 50 38"
    # trajectory = trajectory.split(" ")
    # trajectory = [int(a) for a in trajectory]

    # print(trajectory)

    # state = bridgeplay.construct_state_from_trajectory(trajectory, bridge.default_game)
    # print(state)

    # with open("../belief_sl/exp3/log.txt", "r") as f:
    #     content = f.read()
    # # print(content)

    # log_start = content.find("Epoch")

    # logs = content[log_start:].split("\n")

    # print(logs)

    # loss_list = []
    # acc_list = []
    # for log in logs[:20]:
    #     match = re.match("Epoch (.*?), loss=(.*?), acc=(.*?), model saved=.*?", log)
    #     if match:
    #         print(match.groups())
    #         epoch, loss, acc = match.groups()
    #         loss_list.append(float(loss))
    #         acc_list.append(float(acc))

    # fig, (ax1) = plt.subplots()
    # ax1: plt.Axes

    # lines1,  = ax1.plot(range(0, len(loss_list)), loss_list, label="loss")
    # ax1.set_ylabel("Cross Entropy Loss")
    # ax1.set_xlabel("Epoch")
    # ax1.set_xticks(range(0, len(loss_list), 3))

    # ax2: plt.Axes
    # ax2 = ax1.twinx()

    # lines2,  = ax2.plot(
    #     range(0, len(loss_list)), acc_list, linestyle="--", label="accuracy"
    # )
    # ax2.set_ylabel("Accuracy")
    # lines = [lines1, lines2]
    # labels = [line.get_label() for line in lines]
    # plt.legend(lines, labels, loc="right")

    # plt.show()
