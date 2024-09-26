import math
from collections import OrderedDict

from net import PublicLSTMNet
import torch
import numpy as np
from agent import BridgeLSTMAgent
from pysrc.agent import BridgeFFWDAgent
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

    # a = torch.rand(3, 4)
    # print(a)
    # print(a.sum(0))
    # print(a.sum(0).size())
    # print(a.mean(0).size())
    # print(a.sum(1))
    # print(a.sum(1).size())
    # print(a.mean(1).size())
    # print(a.max(0))
    # print(a.max(1))
    #
    # print(np.round([0.99, 1.01]))
    # p_ckpt = torch.load("ffwd_sl/exp3/p_model1.pthw")
    # # print(p_ckpt["model_state_dict"])
    # # print(type(p_ckpt["model_state_dict"]))
    # for k, v in p_ckpt["model_state_dict"].items():
    #     print(k)
    # v_ckpt = torch.load("ffwd_sl/exp3/v_model2.pthw")
    # # print(v_ckpt["model_state_dict"])
    #
    # agent = BridgeFFWDAgent(
    #     "cuda",
    #     900,
    #     1056,
    #     1024,
    #     1024,
    #     38,
    #     4,
    #     4,
    #     "gelu",
    #     "gelu",
    #     0.0,
    #     "sep"
    # )
    #
    # combined_state_dict = OrderedDict()
    # for k, v in v_ckpt["model_state_dict"].items():
    #     if "v_net" in k or "fc_v" in k:
    #         combined_state_dict.update({k: v})
    #
    # for k, v in p_ckpt["model_state_dict"].items():
    #     if "p_net" in k or "fc_p" in k:
    #         combined_state_dict.update({k: v})
    #
    # print(combined_state_dict)
    # assert combined_state_dict.keys() == p_ckpt["model_state_dict"].keys() == v_ckpt["model_state_dict"].keys()
    # ckpt = {
    #     "model_state_dict": combined_state_dict
    # }
    # torch.save(ckpt, "ffwd_sl/sl_sep.pthw")

    net = torch.nn.Linear(10, 20)
    a = torch.rand(1, 5, 10)
    b = torch.rand(1, 5, 10)
    c = torch.vstack([a, b])
    print(c.shape)
    rep = net(c)
    print(rep.shape)