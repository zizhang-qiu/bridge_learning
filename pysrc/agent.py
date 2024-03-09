from typing import Dict, Optional, Tuple

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
        digits = self.policy_net.forward(obs["s"][:, :480])
        reply = {"pi": torch.nn.functional.softmax(digits, dim=-1)}
        return reply

    @torch.jit.script_method
    def get_belief(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        digits = self.belief_net.forward(obs["s"])
        reply = {"belief": torch.nn.functional.sigmoid(digits)}
        return reply

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        policy = self.get_policy(obs)
        legal_policy = policy["pi"] * obs["legal_move"][:, -38:]
        greedy_a = torch.argmax(legal_policy) + 52
        stochastic_a = torch.multinomial(legal_policy, 1).squeeze() + 52
        reply = {"pi": policy["pi"], "greedy_a": greedy_a, "a": stochastic_a}
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
            "v": value,
        }
        return reply


class SimpleAgent(torch.jit.ScriptModule):
    def __init__(self, policy_conf: Dict):
        super().__init__()
        self.policy_net = MLP.from_conf(policy_conf)

    def act(self, obs: Dict[str, torch.Tensor]):
        policy = torch.nn.functional.softmax(
            self.policy_net(obs["s"].unsqueeze(0)).squeeze(), dim=-1
        )
        policy *= obs["legal_move"]
        a = torch.argmax(policy) + 52
        return a


class BridgeBeliefModel(torch.jit.ScriptModule):
    def __init__(self, conf: Dict, output_size: int):
        super().__init__()
        self.net = MLP.from_conf(conf)
        self.target_key = "belief_he"
        self.pred1 = nn.Linear(self.net.output_size, output_size)
        self.pred2 = nn.Linear(self.net.output_size, output_size)
        self.pred3 = nn.Linear(self.net.output_size, output_size)

    def forward(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        digits = self.net(obs["s"])
        pred1 = torch.nn.functional.sigmoid(self.pred1(digits))
        pred2 = torch.nn.functional.sigmoid(self.pred2(digits))
        pred3 = torch.nn.functional.sigmoid(self.pred3(digits))
        return pred1, pred2, pred3

    def loss(self, batch: Dict[str, torch.Tensor]):
        pred1, pred2, pred3 = self.forward(batch)
        ground_truth = batch[self.target_key]
        bits_per_player = ground_truth.shape[1] // 3
        g_truth1 = ground_truth[:, :bits_per_player]
        g_truth2 = ground_truth[:, bits_per_player : 2 * bits_per_player]
        g_truth3 = ground_truth[:, 2 * bits_per_player :]
        loss_func = nn.BCELoss()
        loss = (
            loss_func(pred1, g_truth1)
            + loss_func(pred2, g_truth2)
            + loss_func(pred3, g_truth3)
        )
        return loss

    def accuracy(self, batch: Dict[str, torch.Tensor]):
        pred1, pred2, pred3 = self.forward(batch)
        ground_truth = batch[self.target_key]
