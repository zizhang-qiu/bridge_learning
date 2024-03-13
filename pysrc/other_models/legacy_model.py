"""
@author: qzz
@contact:q873264077@gmail.com
@file: legacy_model.py
@time: 2024/03/13 10:58
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Final, Union


class PolicyNet(nn.Module):
    def __init__(self):
        """
        A policy net to output policy distribution.
        """
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 38),
        )

    def forward(self, state: torch.Tensor):
        out = self.net(state)
        policy = F.log_softmax(out, -1)
        return policy


class ValueNet(nn.Module):

    def __init__(self):
        """A value net as critic."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(480, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1),
        )

    def forward(self, state: torch.Tensor):
        value = self.net(state)
        return value


class PerfectValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(636, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1),
            nn.Tanh(),
        )

    def forward(self, p_state: torch.Tensor):
        value = self.net(p_state)
        return value


class LegacyModel(torch.jit.ScriptModule):

    def __init__(self, p_net: PolicyNet, perfect=True):
        """
        An agent for acting in vectorized env.
        Args:
            p_net: The policy net.
        """
        super().__init__()
        self.p_net = p_net
        self.perfect = perfect
        self.v_net = PerfectValueNet() if self.perfect else ValueNet()

    @torch.jit.script_method
    def get_probs(self, s: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities for given obs
        Args:
            s (torch.Tensor): The obs tensor, should be 2-dimensional

        Returns:
            torch.Tensor: The probabilities.
        """
        # print("legal_actions", legal_actions)
        probs = torch.exp(self.p_net(s))
        # print("probs", probs)
        return probs

    def get_values(self, s: torch.Tensor):
        values = self.v_net(s)
        return values

    @torch.jit.script_method
    def get_log_probs(self, s: torch.Tensor) -> torch.Tensor:
        probs = self.p_net(s)
        return probs

    @torch.jit.script_method
    def single_act(self, obs: Dict[str, torch.Tensor]):
        greedy = obs["greedy"].item()
        legal_actions = obs["legal_actions"]
        s = obs["s"]
        perfect_s = obs["perfect_s"]
        log_probs = self.get_log_probs(s.unsqueeze(0)).squeeze()
        probs = torch.exp(log_probs)
        probs = probs * legal_actions
        value = (
            self.v_net(perfect_s.unsqueeze(0)).squeeze()
            if self.perfect
            else self.v_net(s.unsqueeze(0)).squeeze()
        )
        # print(probs)
        if greedy:
            action = torch.argmax(probs)
        else:
            if torch.equal(probs, torch.zeros_like(probs)):
                print("Warning: all the probs are zero")
                action = torch.multinomial(legal_actions, 1).squeeze()
            else:
                action = torch.multinomial(probs, 1).squeeze()
        return {
            "a": action.detach().cpu(),
            "log_probs": log_probs.detach().cpu(),
            "values": value.detach().cpu(),
            "raw_probs": probs.detach().cpu(),
        }

    @torch.jit.script_method
    def vec_act(self, obs: Dict[str, torch.Tensor]):
        # vec env obs is always 2d
        s = obs["s"]
        perfect_s = obs["perfect_s"]
        legal_actions = obs["legal_actions"]
        log_probs = self.get_log_probs(s)
        probs = torch.exp(log_probs)
        # topk_values, topk_indices = probs.topk(4, dim=1)
        # mask = torch.zeros_like(probs)
        # mask.scatter_(1, topk_indices, 1)
        legal_probs = probs * legal_actions
        all_zeros = torch.all(legal_probs == 0, dim=1)
        zero_indices = torch.nonzero(all_zeros).squeeze()
        if zero_indices.numel() > 0:
            legal_probs[zero_indices] = legal_actions[zero_indices]
        greedy = obs["greedy"]

        greedy_action = torch.argmax(legal_probs, 1)
        # print("greedy actions",  greedy_action)
        random_actions = torch.multinomial(legal_probs, 1).squeeze(1)

        # print("random actions", random_actions)
        action = greedy * greedy_action + (1 - greedy) * random_actions
        values = self.v_net(perfect_s).squeeze() if self.perfect else self.v_net(s)
        return {
            "a": action.detach().cpu(),
            "log_probs": log_probs.detach().cpu(),
            "values": values.detach().cpu(),
            "raw_probs": probs.detach().cpu(),
        }

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get actions for given obs
        Args:
            obs (Dict[str, torch.Tensor]): The obs tensordict

        Returns:
            Dict[str, torch.Tensor]: The reply contains action, values and probs

        """
        # vec env obs is always 2d
        greedy = obs["greedy"]
        if greedy.numel() > 1:
            reply = self.vec_act(obs)
        else:
            reply = self.single_act(obs)
        return reply

    @torch.jit.script_method
    def act_greedy(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s = obs["s"][:, :480]
        legal_actions = obs["legal_move"][:, -38:]
        log_probs = self.get_log_probs(s)
        probs = torch.exp(log_probs)
        # topk_values, topk_indices = probs.topk(4, dim=1)
        # mask = torch.zeros_like(probs)
        # mask.scatter_(1, topk_indices, 1)
        legal_probs = probs * legal_actions
        all_zeros = torch.all(legal_probs == 0, dim=1)
        zero_indices = torch.nonzero(all_zeros).squeeze()
        if zero_indices.numel() > 0:
            legal_probs[zero_indices] = legal_actions[zero_indices]
        # greedy = 1

        greedy_action = torch.argmax(legal_probs, 1, keepdim=True) + 52
        # print("greedy actions",  greedy_action)
        # random_actions = torch.multinomial(legal_probs, 1).squeeze(1)

        return {
            "a": greedy_action.detach().cpu(),
            "log_probs": log_probs.detach().cpu(),
        }
        
        
def legacy_agent(device:str="cuda") -> LegacyModel:
    policy_net = PolicyNet()
    policy_net.load_state_dict(torch.load("../external_models/legacy/model.pth")["model_state_dict"]["policy"])
    agent = LegacyModel(policy_net)
    agent.to(device)
    return agent
