from __future__ import annotations

from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(num_fc_layers: int, in_dim: int = 480, out_dim: int = 38,
              hid_dim=1024) -> nn.Sequential:
    ff_layers = [nn.Linear(in_dim, hid_dim), nn.GELU()]
    for i in range(1, num_fc_layers):
        ff_layers.append(nn.Linear(hid_dim, hid_dim))
        ff_layers.append(nn.GELU())
    ff_layers.append(nn.Linear(hid_dim, out_dim))
    net = nn.Sequential(*ff_layers)
    return net


class MLP(torch.jit.ScriptModule):
    __constants__ = [
        "num_fc_layers",
        "in_dim",
        "out_dim",
        "hid_dim"
    ]

    def __init__(self, num_fc_layers: int, in_dim: int = 480, out_dim: int = 38,
                 hid_dim=1024):
        super().__init__()
        self.num_fc_layers = num_fc_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim

        self.net = build_mlp(self.num_fc_layers,
                             self.in_dim,
                             self.out_dim,
                             self.hid_dim)

    @torch.jit.script_method
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.net(s)
        return x

    def get_conf(self) -> Dict[str, Union[int, str]]:
        conf = {
            "num_fc_layers": self.num_fc_layers,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "hid_dim": self.hid_dim
        }
        return conf

    def get_save_dict(self):
        conf = self.get_conf()
        state_dict = self.state_dict()
        save_dict = {
            "conf": conf,
            "state_dict": state_dict
        }
        return save_dict

    def save_with_conf(self, save_path: str):
        conf = self.get_conf()
        state_dict = self.state_dict()
        save_dict = {
            "conf": conf,
            "state_dict": state_dict
        }
        torch.save(save_dict, save_path)

    @classmethod
    def from_conf(cls, conf: Dict[str, Union[int, str]]) -> MLP:
        num_fc_layers = conf.get("num_fc_layers", 4)
        in_dim = conf.get("in_dim", 480)
        out_dim = conf.get("out_dim", 38)
        hid_dim = conf.get("hid_dim", 1024)
        return cls(num_fc_layers=num_fc_layers,
                   in_dim=in_dim,
                   out_dim=out_dim,
                   hid_dim=hid_dim)

    @classmethod
    def from_file(cls, file: Union[str, Dict]) -> MLP:
        if isinstance(file, str):
            file = torch.load(file)
        conf = file["conf"]
        state_dict = file["state_dict"]
        net = cls.from_conf(conf)
        net.load_state_dict(state_dict)
        return net
