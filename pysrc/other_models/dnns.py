from typing import Dict
import torch
from torch import nn
from .model_utils import LinearList

class FullyConnectedSkipNN(torch.jit.ScriptModule):
    def __init__(self, input_dim:int, hidden_dim:int, out_dim:int, num_blocks: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.net = LinearList(hidden_dim, num_blocks)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear1(x))
        x = self.net(x)
        out = self.linear2(x)
        return out
        


class DNNsModel(torch.jit.ScriptModule):

    def __init__(
        self,
        num_enn_in_dim: int = 372,
        num_enn_hidden_dim: int = 1500,
        num_enn_out_dim: int = 52,
        num_enn_hidden_blocks: int = 3,
        num_pnn_in_dim: int = 424,
        num_pnn_hidden_dim: int = 1200,
        num_pnn_out_dim: int = 38,
        num_pnn_hidden_blocks: int = 4,
    ):
        super().__init__()
        self.enn = FullyConnectedSkipNN(
            num_enn_in_dim, num_enn_hidden_dim, num_enn_out_dim, num_enn_hidden_blocks
        )
        self.pnn = FullyConnectedSkipNN(
            num_pnn_in_dim, num_pnn_hidden_dim, num_pnn_out_dim, num_pnn_hidden_blocks
        )

        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(-1)

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The key is set to dnns_s
        x = obs["dnns_s"]

        hand_estimation = self.sigmoid(self.enn(x))

        pnn_s = torch.cat((x, hand_estimation), dim=1)

        pi = self.softmax(self.pnn(pnn_s))

        return {"pi": pi.detach(), "p_hand": hand_estimation.detach()}
    
    @torch.jit.script_method
    def act_greedy(self, obs:Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        reply = self.act(obs)
        a = reply["pi"].multinomial(1, replacement=True)
        reply["a"] = a
        return reply
