from typing import Dict, Tuple

import torch
from torch import nn

from net import MLP, PublicLSTMNet, LSTMNet
from set_path import append_sys_path

append_sys_path()
import pyrela


class BridgeA2CModel(torch.jit.ScriptModule):

    def __init__(self, policy_conf: Dict, value_conf: Dict, belief_conf: Dict):
        super().__init__()
        self.policy_net = MLP.from_conf(policy_conf)
        self.value_net = MLP.from_conf(value_conf)
        self.belief_net = MLP.from_conf(belief_conf)
        self.policy_net.eval()
        self.value_net.eval()
        self.belief_net.eval()

    @torch.jit.script_method
    def get_policy(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # if obs["s"]
        digits = self.policy_net.forward(obs["s"][:, :480])
        reply = {"pi": torch.nn.functional.softmax(digits, dim=-1)}
        return reply

    @torch.jit.script_method
    def get_belief(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        digits = self.belief_net.forward(obs["s"][:, :480])
        reply = {"belief": torch.nn.functional.sigmoid(digits)}
        return reply

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        policy = self.get_policy(obs)
        legal_policy = policy["pi"] * obs["legal_move"][:, -38:]
        greedy_a = legal_policy.max(dim=1)[1].view(-1, 1) + 52
        stochastic_a = torch.multinomial(legal_policy, 1, replacement=True) + 52
        reply = {"pi": policy["pi"], "greedy_a": greedy_a, "a": stochastic_a}
        # print(reply)
        return reply

    @torch.jit.script_method
    def act_greedy(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        policy = self.get_policy(obs)
        legal_policy = policy["pi"] * obs["legal_move"][:, -38:]
        greedy_a = legal_policy.max(dim=1)[1].view(-1, 1) + 52
        return {"a": greedy_a}

    @torch.jit.script_method
    def compute_priority(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batchsize = obs["s"].size(0)
        # print(f"Computing priority, batchsize: {batchsize}")
        priority = torch.ones(batchsize, 1)
        return {"priority": priority}

    def clone(self, device: str):
        policy_conf = self.policy_net.get_conf()
        value_conf = self.value_net.get_conf()
        belief_conf = self.belief_net.get_conf()
        new_model = BridgeA2CModel(policy_conf, value_conf, belief_conf)
        new_model.policy_net.load_state_dict(self.policy_net.state_dict())
        new_model.value_net.load_state_dict(self.value_net.state_dict())
        new_model.belief_net.load_state_dict(self.belief_net.state_dict())
        new_model.to(device)
        return new_model


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
        g_truth2 = ground_truth[:, bits_per_player: 2 * bits_per_player]
        g_truth3 = ground_truth[:, 2 * bits_per_player:]
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


class BridgeLSTMAgent(torch.jit.ScriptModule):
    __constants__ = ["in_dim", "hid_dim", "out_dim",
                     "num_priv_mlp_layer", "num_publ_mlp_layer", "num_lstm_layer", "greedy"]

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
            dropout: float = 0.,
            net: str = "publ-lstm",
            greedy=False,
    ):
        super().__init__()
        if net == "publ-lstm":
            self.net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_priv_mlp_layer, num_publ_mlp_layer, num_lstm_layer, activation,
                dropout
            ).to(device)
        elif net == "lstm":
            self.net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_priv_mlp_layer, num_publ_mlp_layer, num_lstm_layer, activation,
                dropout
            ).to(device)
        else:
            raise ValueError(f"The net type {net} is not supported.")
        self.greedy = greedy
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_priv_mlp_layer = num_priv_mlp_layer
        self.num_publ_mlp_layer = num_publ_mlp_layer
        self.num_lstm_layer = num_lstm_layer
        self.activation = activation

    @torch.jit.script_method
    def get_h0(self) -> Dict[str, torch.Tensor]:
        return self.net.get_h0()

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        priv_s = obs["priv_s"]
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        hid = {"h0": obs["h0"], "c0": obs["c0"]}
        reply = self.net.act(priv_s, publ_s, hid)

        # print("get_reply")
        legal_pi = reply["pi"] * legal_move[:, -self.out_dim:]
        # legal_pi[legal_move[:, -self.out_dim:] == 1] += 1e-8
        if self.greedy:
            action = legal_pi.max(dim=1)[1].view(-1, 1) + 52
        else:
            action = torch.multinomial(legal_pi, 1).squeeze().view(-1, 1) + 52

        ret = {
            "a": action,
            "pi": reply["pi"],
            "v": reply["v"],
            "h0": reply["h0"],
            "c0": reply["c0"],
        }
        # print(ret["a"])
        return ret

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "h0" in obs.keys():
            hid = {"h0": obs["h0"], "c0": obs["c0"]}
        else:
            hid = {}
        return self.net.forward(obs["priv_s"], obs["publ_s"], obs["legal_move"], hid)

    def compute_loss_and_priority(self,
                                  batch: pyrela.RNNTransition,
                                  clip_eps: float,
                                  entropy_ratio: float,
                                  value_loss_weight: float):
        priv_s = batch.obs["priv_s"]
        seq_len, batch_size, _ = priv_s.size()
        mask = torch.arange(0, priv_s.size(0), device=batch.seq_len.device)
        # [seq_len, batch_size]
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        # print(mask.size())
        # print("mask: ", mask)
        reply = self.forward(batch.obs)
        pi = torch.nn.functional.softmax(reply["pi"], dim=-1)
        # [seq_len, batch_size, 1]
        v = reply["v"]

        action = batch.action["a"]
        action[action >= 52] -= 52

        mask[action.squeeze(2) == self.out_dim - 1] = 0
        # print("mask: ", mask)

        # [seq_len, batch_size, num_actions]
        current_log_probs = torch.log(pi + 1e-8)
        # current_log_probs = current_log_probs * mask
        # [seq_len * batch_size, num_actions]
        # current_log_probs = current_log_probs.view(-1, self.out_dim)

        # print("current_log_probs: ", current_log_probs)

        current_action_log_probs = current_log_probs.gather(2, action).squeeze(2)

        # print("current_action_log_probs: ", current_action_log_probs)
        old_probs = batch.action["pi"]
        old_log_probs = torch.log(old_probs + 1e-8)
        old_action_log_probs = old_log_probs.gather(
            2,
            action.long()
        ).squeeze(2)
        # print("old_action_log_probs: ", old_action_log_probs)

        ratio = torch.exp(current_action_log_probs - old_action_log_probs)
        # print("ratio: ", ratio)

        # [seq_len, batch_size]
        advantage = (batch.reward / 7600 - v.squeeze()) * mask
        # print("advantage: ", advantage)

        surr1 = ratio * (advantage.detach())
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (advantage.detach())
        # print("surr1: ", surr1)
        # print("surr2: ", surr2)
        entropy = -torch.sum(pi * current_log_probs, dim=-1)
        # print("entropy: ", entropy)

        policy_loss = -torch.min(surr1, surr2) - entropy_ratio * entropy
        policy_loss = policy_loss * mask

        value_loss = torch.pow(advantage, 2) * value_loss_weight
        value_loss = value_loss * mask

        priority = advantage.detach() * mask

        return policy_loss.mean(0), value_loss.mean(0), priority.sum(0).abs().cpu()
