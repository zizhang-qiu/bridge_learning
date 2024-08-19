from __future__ import annotations

from typing import Dict, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils import activation_function_from_str


def create_mlp(
        input_size: int,
        output_size: int,
        num_hidden_layers: int,
        hidden_size: int,
        activation_function: str,
        activation_args: Optional[Dict] = None,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        use_layer_norm: bool = False,
) -> nn.Module:
    """
    Create a Multi-Layer Perceptron (MLP) model.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        num_hidden_layers (int): Number of hidden layers.
        hidden_size (int): Number of neurons in each hidden layer.
        activation_function (Union[nn.Module, str]): Activation function instance.
        activation_args (Optional[Dict]): Additional parameters for the activation function.
        use_dropout (bool): Whether to use Dropout.
        dropout_prob (float): Dropout probability.
        use_layer_norm (bool): Whether to use Layer Normalization.

    Returns:
        nn.Module: Constructed MLP model.
    """
    layers = []
    if isinstance(activation_function, str):
        # activation_function = getattr(torch.nn, activation_function)

        activation_func = activation_function_from_str(activation_function)
    else:
        activation_func = activation_function

    # Add input layer to the first hidden layer
    layers.append(nn.Linear(input_size, hidden_size))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_size))
    layers.append(
        activation_func(**activation_args) if activation_args else activation_func()
    )
    if use_dropout:
        layers.append(nn.Dropout(p=dropout_prob))

    # Add middle hidden layers
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(
            activation_func(**activation_args) if activation_args else activation_func()
        )
        if use_dropout:
            layers.append(nn.Dropout(p=dropout_prob))

    # Add the last hidden layer to the output layer
    layers.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*layers)


class MLP(torch.jit.ScriptModule):
    __constants__ = [
        "input_size",
        "output_size",
        "num_hidden_layers",
        "hidden_size",
        "use_dropout",
        "dropout_prob",
        "use_layer_norm",
    ]

    def __init__(
            self,
            input_size: int = 480,
            output_size: int = 38,
            num_hidden_layers: int = 4,
            hidden_size: int = 1024,
            activation_function: str = "gelu",
            activation_args: Optional[Dict] = None,  # type: ignore
            use_dropout: bool = False,
            dropout_prob: float = 0.5,
            use_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.activation_args = activation_args
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.use_layer_norm = use_layer_norm

        self.net = create_mlp(
            input_size=self.input_size,
            output_size=self.output_size,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            activation_function=self.activation_function,
            activation_args=self.activation_args,
            use_dropout=self.use_dropout,
            dropout_prob=self.dropout_prob,
            use_layer_norm=self.use_layer_norm,
        )

    @torch.jit.script_method
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.net(s)
        return x

    def get_conf(self) -> Dict[str, Union[int, bool, float, Dict, None]]:
        conf = dict(
            input_size=self.input_size,
            output_size=self.output_size,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            activation_function=self.activation_function,
            activation_args=self.activation_args,
            use_dropout=self.use_dropout,
            dropout_prob=self.dropout_prob,
            use_layer_norm=self.use_layer_norm,
        )
        return conf  # type: ignore

    def get_save_dict(self):
        conf = self.get_conf()
        state_dict = self.state_dict()
        save_dict = {"conf": conf, "state_dict": state_dict}
        return save_dict

    def save_with_conf(self, save_path: str):
        conf = self.get_conf()
        state_dict = self.state_dict()
        save_dict = {"conf": conf, "state_dict": state_dict}
        torch.save(save_dict, save_path)

    @classmethod
    def from_conf(cls, conf: Dict[str, Union[int, bool, Dict, nn.Module, str]]) -> MLP:
        return cls(**conf)  # type: ignore

    @classmethod
    def from_file(cls, file: Union[str, Dict]) -> MLP:
        if isinstance(file, str):
            file = torch.load(file)
        conf: Dict = file["conf"]  # type: ignore

        net = cls.from_conf(conf)
        if "state_dict" in conf.keys():
            state_dict = file["state_dict"]  # type: ignore
            net.load_state_dict(state_dict)  # type: ignore
        return net



def get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "softmax":
        return nn.Softmax()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError


class LSTMNet(torch.jit.ScriptModule):
    __constants__ = ["in_dim", "hid_dim", "out_dim", "num_priv_mlp_layer", "num_publ_mlp_layer", "num_lstm_layer"]
    def __init__(self,
                 device: str,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 num_priv_mlp_layer: int,
                 num_publ_mlp_layer: int,
                 num_lstm_layer: int,
                 activation:str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_priv_mlp_layer = num_priv_mlp_layer
        self.num_publ_mlp_layer = num_publ_mlp_layer
        self.num_lstm_layer = num_lstm_layer
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)

        self.lstm.flatten_parameters()
        self.activation = get_activation(activation)
        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), self.activation]
        for i in range(1, self.num_priv_mlp_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(self.activation)
        self.net = nn.Sequential(*ff_layers)
        self.policy_head = nn.Linear(hid_dim, out_dim)
        self.value_head = nn.Linear(hid_dim, 1)

    @torch.jit.script_method
    def get_h0(self) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
            self, priv_s: torch.Tensor, publ_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert priv_s.dim() == 2
        batch_size = hid["h0"].size(0)
        assert hid["h0"].dim() == 3
        # hid size: [batch, num_layer, dim]
        # -> [num_layer, batch, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).contiguous(),
            "c0": hid["c0"].transpose(0, 1).contiguous(),
        }
        priv_s = priv_s.unsqueeze(0)

        x = self.net(priv_s)

        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        o = o.squeeze(0)

        pi = nn.functional.softmax(self.policy_head(o), dim=-1)
        v = self.value_head(o)

        interim_hid_shape = (
            self.num_lstm_layer,
            batch_size,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return {"pi": pi, "v": v, "h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
            self,
            priv_s: torch.Tensor,
            publ_s: torch.Tensor,
            legal_move: torch.Tensor,
            hid: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        assert (
                priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            one_step = True

        x = self.net(priv_s)
        if len(hid) == 0:
            o, _ = self.lstm(x)
        else:
            o, _ = self.lstm(x, (hid["h0"], hid["c0"]))

        o = self.dropout(o)
        pi = self.policy_head(o)
        v = self.value_head(o)
        legal_pi = pi * legal_move[:, :, -self.out_dim:]

        if one_step:
            pi = pi.squeeze(0)
            v = v.squeeze(0)
            legal_pi = legal_pi.squeeze(0)
        return {"pi": pi, "v": v, "legal_pi": legal_pi}


class PublicLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["in_dim", "hid_dim", "out_dim", "num_priv_mlp_layer", "num_publ_mlp_layer", "num_lstm_layer"]

    # class PublicLSTMNet(nn.Module):
    def __init__(
            self,
            device: str,
            in_dim: int,
            hid_dim: int,
            out_dim: int,
            num_priv_mlp_layer: int,
            num_publ_mlp_layer: int,
            num_lstm_layer: int,
            activation: str,
            dropout: float = 0.
    ):
        super().__init__()
        self.device = device
        self.in_dim = in_dim
        self.priv_in_dim = in_dim
        self.publ_in_dim = in_dim - 52

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_priv_mlp_layer = num_priv_mlp_layer
        self.num_publ_mlp_layer = num_publ_mlp_layer
        self.num_lstm_layer = num_lstm_layer
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

        ff_layers = [nn.Linear(self.priv_in_dim, self.hid_dim), self.activation]
        for i in range(1, self.num_priv_mlp_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(self.activation)
        self.priv_net = nn.Sequential(*ff_layers)

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), self.activation]
        for i in range(1, self.num_publ_mlp_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(self.activation)
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.policy_head = nn.Linear(self.hid_dim, out_dim)
        self.value_head = nn.Linear(self.hid_dim, 1)

    @torch.jit.script_method
    def get_h0(self) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
            self, priv_s: torch.Tensor, publ_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert priv_s.dim() == 2

        batch_size = hid["h0"].size(0)
        assert hid["h0"].dim() == 3
        # hid size: [batch, num_layer, dim]
        # -> [num_layer, batch, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).contiguous(),
            "c0": hid["c0"].transpose(0, 1).contiguous(),
        }

        priv_s = priv_s.unsqueeze(0)
        publ_s = publ_s.unsqueeze(0)

        priv_o = self.priv_net(priv_s)
        x = self.publ_net(publ_s)
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        # print(h, h.size(), c, c.size())

        o = priv_o * publ_o
        o = o.squeeze(0)
        pi = torch.nn.functional.softmax(self.policy_head(o), dim=-1)
        v = self.value_head(o)

        interim_hid_shape = (
            self.num_lstm_layer,
            batch_size,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return {"pi": pi, "v": v, "h0": h, "c0": c}

    def forward(
            self,
            priv_s: torch.Tensor,
            publ_s: torch.Tensor,
            legal_move: torch.Tensor,
            hid: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        assert (
                priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            one_step = True

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        o = self.dropout(o)
        # pi = torch.nn.functional.softmax(self.policy_head(o), dim=-1)
        pi = self.policy_head(o)
        v = self.value_head(o)
        legal_pi = pi * legal_move[:, :, -self.out_dim:]

        if one_step:
            pi = pi.squeeze(0)
            v = v.squeeze(0)
            legal_pi = legal_pi.squeeze(0)
        return {"pi": pi, "v": v, "legal_pi": legal_pi}
