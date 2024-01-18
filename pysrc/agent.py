from typing import Dict, Optional

import torch
from torch import nn

from net import MLP


class BridgeA2CModel(torch.jit.ScriptModule):

    def __init__(self, policy_conf: Dict, value_conf: Dict, belief_conf: Dict):
        super().__init__()
        self.policy_net = MLP.from_conf(policy_conf)
        self.value_net = MLP.from_conf(value_conf)
        self.belief_net = MLP.from_conf(belief_conf)

    @torch.jit.script_method
    def get_policy(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # if obs["s"]
        digits = self.policy_net.forward(obs["s"])
        reply = {
            "pi": torch.nn.functional.softmax(digits, dim=-1)
        }
        return reply

    @torch.jit.script_method
    def get_belief(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        digits = self.belief_net.forward(obs["s"])
        reply = {
            "belief": torch.nn.functional.sigmoid(digits)
        }
        return reply

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        policy = self.get_policy(obs)
        legal_policy = policy["pi"] * obs["legal_moves"]
        greedy_a = torch.argmax(legal_policy) + 52
        stochastic_a = torch.multinomial(legal_policy, 1).squeeze() + 52
        reply = {
            "pi": policy["pi"],
            "greedy_a": greedy_a,
            "a": stochastic_a
        }
        return reply

    @torch.jit.script_method
    def compute_priority(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batchsize = obs["s"].size(0)
        # print(f"Computing priority, batchsize: {batchsize}")
        priority = torch.ones(batchsize, 1)
        return {"priority": priority}


class BridgeAgent(torch.jit.ScriptModule):

    def __init__(self, policy_conf: Dict, value_conf: Dict):
        super().__init__()
        self.policy_net = MLP.from_conf(policy_conf)
        self.value_net = MLP.from_conf(value_conf)

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        policy = torch.nn.functional.softmax(self.policy_net(obs["s"]))
        policy *= obs["legal_move"]
        value = self.value_net(obs["s"]).squeeze()
        greedy_action = torch.argmax(policy, 1) + 52
        stochastic_action = torch.multinomial(policy, 1).squeeze() + 52
        reply = {
            "a": stochastic_action.to(torch.int32),
            "g_a": greedy_action.to(torch.int32),
            "v": value
        }
        return reply


class SimpleAgent(torch.jit.ScriptModule):
    def __init__(self, policy_conf: Dict):
        super().__init__()
        self.policy_net = MLP.from_conf(policy_conf)

    def act(self, obs: Dict[str, torch.Tensor]):
        policy = torch.nn.functional.softmax(self.policy_net(obs["s"].unsqueeze(0)).squeeze(), dim=-1)
        policy *= obs["legal_move"]
        a = torch.argmax(policy) + 52
        return a
