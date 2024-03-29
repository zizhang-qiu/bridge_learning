from typing import Dict
import torch
import torch.jit
from torch import nn

class PIModel(torch.jit.ScriptModule):
    """Model used in 'Human-Agent Cooperation in Bridge Bidding'."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 38),
            nn.Softmax(-1),
        )

    @torch.jit.script_method
    def act(self, obs:Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        pi = self.net(obs["s"][:, :480])
        legal_pi = pi * obs["legal_move"][:, -38:]

        greedy_a = torch.argmax(legal_pi) + 52
        stochastic_a = torch.multinomial(legal_pi, 1).squeeze() + 52

        return {"pi": pi, "greedy_a": greedy_a, "a": stochastic_a}

    @torch.jit.script_method
    def act_greedy(self, obs:Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        pi = self.net(obs["s"][:, :480])
        legal_pi = pi * obs["legal_move"][:, -38:]

        greedy_a = torch.argmax(legal_pi) + 52
        return {"pi":pi, "a":greedy_a}
