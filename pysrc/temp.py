from net import PublicLSTMNet
import torch
import numpy as np
from agent import BridgePublicLSTMagent
from utils import load_dataset

if __name__ == "__main__":
    # net = PublicLSTMNet("cuda", 480, 1024, 48, 4, 1)
    # net.to("cuda")
    # print(net)
    # agent = BridgePublicLSTMagent("cuda", 480, 1024, 38, 4, 1)

    # batch_size = 4
    # priv_s = torch.rand(size=(batch_size, 52)).to("cuda")
    # publ_s = torch.rand(size=(batch_size, 428)).to("cuda")
    # hid = agent.net.get_h0()
    # h0 = torch.zeros(1, 1024)
    # # h0 = torch.repeat_interleave(h0, batch_size, 0).reshape(batch_size, 1, 1024)
    # # print(h0.size())
    # hid = {
    #     k: torch.repeat_interleave(v, batch_size, 0)
    #     .reshape(batch_size, 1, 1024)
    #     .to("cuda")
    #     for k, v in hid.items()
    # }
    # print(hid)
    # obs = {
    #     "priv_s": priv_s,
    #     "publ_s": publ_s,
    #     "h0": hid["h0"],
    #     "c0": hid["c0"],
    #     "legal_move": torch.ones(batch_size, 38).to("cuda"),
    # }

    # reply = agent.act(obs)
    # print(reply)
    
    dataset = load_dataset(r"D:\Projects\bridge_research\expert\test.txt")
    lengths = [len(data) - 52 * 2 if len(data) > 52 + 4 else len(data) - 52
               for data in dataset]
    max_len = max(lengths)
    min_len = min(lengths)
    
    print(max_len, min_len)
    