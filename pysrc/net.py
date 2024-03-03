from __future__ import annotations

from typing import Dict, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils import activation_function_from_str


def create_mlp(input_size: int, output_size: int, num_hidden_layers: int, hidden_size: int,
               activation_function: Union[nn.Module, str], activation_args: Optional[Dict] = None,
               use_dropout: bool = False,
               dropout_prob: float = 0.5, use_layer_norm: bool = False) -> nn.Module:
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
        activation_function = activation_function_from_str(activation_function)

    # Add input layer to the first hidden layer
    layers.append(nn.Linear(input_size, hidden_size))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_size))
    layers.append(activation_function(**activation_args) if activation_args else activation_function())
    if use_dropout:
        layers.append(nn.Dropout(p=dropout_prob))

    # Add middle hidden layers
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation_function(**activation_args) if activation_args else activation_function())
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
        "use_layer_norm"
    ]

    def __init__(self, input_size: int = 480, output_size: int = 38, num_hidden_layers: int = 4,
                 hidden_size: int = 1024,
                 activation_function: nn.Module = nn.ReLU, activation_args: Optional[Dict] = None, # type: ignore
                 use_dropout: bool = False,
                 dropout_prob: float = 0.5, use_layer_norm: bool = False):
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

        self.net = create_mlp(input_size=self.input_size,
                              output_size=self.output_size,
                              num_hidden_layers=self.num_hidden_layers,
                              hidden_size=self.hidden_size,
                              activation_function=self.activation_function,
                              activation_args=self.activation_args,
                              use_dropout=self.use_dropout,
                              dropout_prob=self.dropout_prob,
                              use_layer_norm=self.use_layer_norm)

    @torch.jit.script_method
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.net(s)
        return x

    def get_conf(self) -> Dict[str, Union[int, bool, Dict, nn.Module]]:
        conf = dict(input_size=self.input_size,
                    output_size=self.output_size,
                    num_hidden_layers=self.num_hidden_layers,
                    hidden_size=self.hidden_size,
                    activation_function=self.activation_function,
                    activation_args=self.activation_args,
                    use_dropout=self.use_dropout,
                    dropout_prob=self.dropout_prob,
                    use_layer_norm=self.use_layer_norm)
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
    def from_conf(cls, conf: Dict[str, Union[int, bool, Dict, nn.Module, str]]) -> MLP:
        return cls(**conf)

    @classmethod
    def from_file(cls, file: Union[str, Dict]) -> MLP:
        if isinstance(file, str):
            file = torch.load(file)
        conf: Dict = file["conf"]

        net = cls.from_conf(conf)
        if "state_dict" in conf.keys():
            state_dict = file["state_dict"]
            net.load_state_dict(state_dict)
        return net
