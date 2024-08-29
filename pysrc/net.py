from __future__ import annotations

from typing import Dict, Union, Optional

import torch
import torch.nn as nn

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


class FFWDA2CWeightSharingNet(torch.jit.ScriptModule):
    __constants__ = ["in_dim", "hid_dim", "out_dim", "num_mlp_layer", "dropout"]

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 num_mlp_layer: int,
                 activation: str,
                 dropout: float):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_mlp_layer = num_mlp_layer
        self.activation = get_activation(activation)
        self.dropout = dropout
        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), self.activation]
        for i in range(1, self.num_mlp_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(self.activation)
        self.net = nn.Sequential(*ff_layers)
        self.fc_p = nn.Linear(hid_dim, out_dim)
        self.fc_v = nn.Linear(hid_dim, 1)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        single_act = False
        if priv_s.dim() == 1:
            single_act = True
            priv_s = priv_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)

        x = self.net(priv_s)

        x = torch.nn.functional.dropout(x, self.dropout, self.training)
        logits = self.fc_p(x)
        v = self.fc_v(x)
        pi = torch.nn.functional.softmax(logits, dim=-1)
        legal_pi = pi * legal_move

        if single_act:
            pi = pi.squeeze(0)
            v = v.squeeze(0)
            legal_pi = legal_pi.squeeze(0)

        return {"pi": pi, "v": v, "legal_pi": legal_pi}


class FFWDA2CSeparateNet(torch.jit.ScriptModule):
    __constants__ = ["p_in_dim", "v_in_dim"]

    def __init__(self,
                 p_in_dim: int,
                 v_in_dim: int,
                 p_hid_dim: int,
                 v_hid_dim: int,
                 p_out_dim: int,
                 num_p_mlp_layer: int,
                 num_v_mlp_layer: int,
                 p_activation: str,
                 v_activation: str,
                 dropout: float):
        """
        An A2C net using separate policy network and value network, allowing us to use different features for them.
        Args:
            p_in_dim: The input dimension of policy network.
            v_in_dim: The input dimension of value network.
            p_hid_dim: The hidden dimension of policy network.
            v_hid_dim: The hidden dimension of value network.
            p_out_dim: The output dimension of policy network.
            num_p_mlp_layer: The number of layer of policy network.
            num_v_mlp_layer: The number of layer of value network.
            p_activation: The activation function of policy network.
            v_activation: The activation function of value network.
            dropout: The dropout prob.
        """
        super().__init__()
        self.p_in_dim = p_in_dim
        self.v_in_dim = v_in_dim
        self.p_hid_dim = p_hid_dim
        self.v_hid_dim = v_hid_dim
        self.p_out_dim = p_out_dim
        self.num_p_mlp_layer = num_p_mlp_layer
        self.num_v_mlp_layer = num_v_mlp_layer
        self.p_activation = get_activation(p_activation)
        self.v_activation = get_activation(v_activation)
        self.dropout = dropout

        p_ff_layers = [nn.Linear(self.p_in_dim, self.p_hid_dim), self.p_activation]
        for i in range(1, self.num_p_mlp_layer):
            p_ff_layers.append(nn.Linear(self.p_hid_dim, self.p_hid_dim))
            p_ff_layers.append(self.p_activation)

        self.p_net = nn.Sequential(*p_ff_layers)
        self.fc_p = nn.Linear(self.p_hid_dim, self.p_out_dim)

        v_ff_layers = [nn.Linear(self.v_in_dim, self.v_hid_dim), self.v_activation]
        for i in range(1, self.num_v_mlp_layer):
            v_ff_layers.append(nn.Linear(self.v_hid_dim, self.v_hid_dim))
            v_ff_layers.append(self.v_activation)

        self.v_net = nn.Sequential(*v_ff_layers)
        self.fc_v = nn.Linear(self.v_hid_dim, 1)  # Value network always output 1 value.

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        if self.p_in_dim != self.v_in_dim:
            assert "perf_s" in obs
            value_s = obs["perf_s"]
        else:
            value_s = priv_s
        single_act = False
        if priv_s.dim() == 1:
            single_act = True
            priv_s = priv_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            value_s = value_s.unsqueeze(0)

        p_x = self.p_net(priv_s)
        v_x = self.v_net(value_s)

        p_x = torch.nn.functional.dropout(p_x, self.dropout, self.training)
        v_x = torch.nn.functional.dropout(v_x, self.dropout, self.training)

        logits = self.fc_p(p_x)
        pi = torch.nn.functional.softmax(logits, dim=-1)
        legal_pi = pi * legal_move
        v = self.fc_v(v_x)

        if single_act:
            pi = pi.squeeze(0)
            v = v.squeeze(0)
            legal_pi = legal_pi.squeeze(0)

        return {"pi": pi, "v": v, "legal_pi": legal_pi}


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
                 activation: str,
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
        # v = torch.nn.functional.tanh(v)

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
        # v = torch.nn.functional.tanh(v)
        legal_pi = pi * legal_move[:, :, -self.out_dim:]

        if one_step:
            pi = pi.squeeze(0)
            v = v.squeeze(0)
            legal_pi = legal_pi.squeeze(0)
        return {"pi": pi, "v": v, "legal_pi": legal_pi}
