import math

from net import PublicLSTMNet
import torch
import numpy as np
from agent import BridgeLSTMAgent
from utils import load_dataset
from set_path import append_sys_path
import multiprocessing as mp

append_sys_path()
# import sys
# sys.path.append("../cmake-build-release-visual-studio-1")
import bridge
import pyrela
import bridgelearn

if __name__ == "__main__":
    # agent = BridgeLSTMAgent("cuda", 900, 1024, 39,
    #                         4, 2, 1,
    #                         "gelu", 0.0).to("cuda")
    # env = bridgelearn.DuplicateEnv()
    # print(bridgelearn.registered_encoders())
    # a = torch.rand(size=[3, 4, 5])
    # for dim in [-1, 0, 1, 2]:
    #     print(torch.nn.functional.softmax(a, dim))

    a = torch.rand(3, 4)
    print(a)
    print(a.sum(0))
    print(a.sum(0).size())
    print(a.mean(0).size())
    print(a.sum(1))
    print(a.sum(1).size())
    print(a.mean(1).size())
    print(a.max(0))
    print(a.max(1))

    print(np.round([0.99, 1.01]))

