import torch
import torch.nn as nn


def build_mlp(
        *,
        n_in,
        n_hidden,
        n_layers,
        out_size=None,
        act=None,
        use_layer_norm=False,
        dropout=0,
):
    if act is None:
        act = GELU()
    build_norm_layer = (
        lambda: nn.LayerNorm(n_hidden) if use_layer_norm else nn.Sequential()
    )
    build_dropout_layer = (
        lambda: nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
    )

    last_size = n_in
    vals_net = []
    for _ in range(n_layers):
        vals_net.extend(
            [
                nn.Linear(last_size, n_hidden),
                build_norm_layer(),
                act,
                build_dropout_layer(),
            ]
        )
        last_size = n_hidden
    if out_size is not None:
        vals_net.append(nn.Linear(last_size, out_size))
    return nn.Sequential(*vals_net)


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)


class PolicyNet(torch.jit.ScriptModule):
    __constants__ = ["in_dim", "hid_dim", "out_dim", "num_layers"]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.net = build_mlp(n_in=in_dim,
                             n_hidden=hid_dim,
                             n_layers=num_layers,
                             out_size=out_dim,
                             use_layer_norm=True)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        logits = self.net(s)
        softmax_logits = nn.functional.softmax(logits, dim=-1)
        return softmax_logits

