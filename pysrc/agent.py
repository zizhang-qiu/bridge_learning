from typing import Dict, Tuple, Optional

import torch
from torch import nn

from net import MLP, PublicLSTMNet, LSTMNet, FFWDA2CSeparateNet, FFWDA2CWeightSharingNet
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


class BridgeLSTMAgent(torch.jit.ScriptModule):
    __constants__ = [
        "in_dim",
        "hid_dim",
        "out_dim",
        "num_priv_mlp_layer",
        "num_publ_mlp_layer",
        "num_lstm_layer",
        "uniform_priority",
    ]

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
        dropout: float = 0.0,
        net: str = "publ-lstm",
        greedy=False,
        uniform_priority=True,
    ):
        super().__init__()
        if net == "publ-lstm":
            self.network = PublicLSTMNet(
                device,
                in_dim,
                hid_dim,
                out_dim,
                num_priv_mlp_layer,
                num_publ_mlp_layer,
                num_lstm_layer,
                activation,
                dropout,
            ).to(device)
        elif net == "lstm":
            self.network = LSTMNet(
                device,
                in_dim,
                hid_dim,
                out_dim,
                num_priv_mlp_layer,
                num_publ_mlp_layer,
                num_lstm_layer,
                activation,
                dropout,
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
        self.dropout = dropout
        self.activation = activation
        self.net = net
        self.uniform_priority = uniform_priority

    def clone(self, device: str, overwrite: Optional[Dict] = None):

        if overwrite is None:
            overwrite = dict()
        cloned = type(self)(
            device,
            self.in_dim,
            self.hid_dim,
            self.out_dim,
            self.num_priv_mlp_layer,
            self.num_publ_mlp_layer,
            self.num_lstm_layer,
            overwrite.get("activation", self.activation),
            overwrite.get("dropout", self.dropout),
            self.net,
            overwrite.get("greedy", self.greedy),
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    @torch.jit.script_method
    def get_h0(self) -> Dict[str, torch.Tensor]:
        return self.network.get_h0()

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        priv_s = obs["priv_s"]
        # print(priv_s.size())
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        hid = {"h0": obs["h0"], "c0": obs["c0"]}
        reply = self.network.act(priv_s, publ_s, hid)

        # print("get_reply")
        # for k, v in reply.items():
        #     print(k, v.size())
        legal_pi = reply["pi"] * legal_move[:, -self.out_dim :]
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
        return self.network.forward(
            obs["priv_s"], obs["publ_s"], obs["legal_move"], hid
        )

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Uniform priority, all to 1.
        # Since the input is batched in 0 dim, we have to use size(0) here.
        if self.uniform_priority:
            return {"priority": torch.ones(input_["reward"].size(0))}

        for k, v in input_.items():
            if v.dim() > 1:
                input_[k] = v.transpose(0, 1).contiguous()
        reply = self.forward(input_)

        v = reply["v"]

        # [seq_len, batch_size]
        advantage = input_["reward"] - v.squeeze()
        priority = advantage.abs()
        priority = priority.sum(0) / input_["seq_len"]
        return {"priority": priority}

    def compute_loss_and_priority(
        self,
        batch: pyrela.RNNTransition,
        clip_eps: float,
        entropy_ratio: float,
        value_loss_weight: float,
        policy_no_op_coeff: float,
        value_no_op_coeff: float,
    ):
        priv_s = batch.obs["priv_s"]
        seq_len, batch_size, _ = priv_s.size()
        mask = torch.arange(0, seq_len, device=batch.seq_len.device)
        # [seq_len, batch_size]
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        # print(mask.size())
        # print("mask: ", mask)
        batch.obs["h0"] = batch.h0["h0"]
        batch.obs["c0"] = batch.h0["c0"]
        reply = self.forward(batch.obs)
        # [seq_len, batch_size, num_actions]
        pi = torch.nn.functional.softmax(reply["pi"], dim=-1)
        # print("pi: ", pi)
        # print("old pi: ", batch.action["pi"])
        # [seq_len, batch_size, 1]
        v = reply["v"]

        action = batch.action["a"]
        action[action >= 52] -= 52
        # action = action[mask == 1]
        # print("action", action.squeeze(2))

        policy_mask = mask.clone()
        policy_mask[action.squeeze(2) == self.out_dim - 1] = policy_no_op_coeff
        value_mask = mask.clone()
        value_mask[action.squeeze(2) == self.out_dim - 1] = value_no_op_coeff
        # mask[action.squeeze(2) == self.out_dim - 1] = 0
        # print("mask: ", mask)

        # pi = pi[mask == 1]
        # v = v[mask == 1]
        # print("pi: ", pi)
        # print("v: ", v)

        # weight = torch.repeat_interleave(batch_weight.unsqueeze(0), seq_len, 0)
        # weight = weight[mask == 1]

        # [seq_len, batch_size, num_actions]
        current_log_probs = torch.log(pi + 1e-16)
        # [seq_len * batch_size, num_actions]
        # current_log_probs = current_log_probs.view(-1, self.out_dim)

        # print("current_log_probs: ", current_log_probs)

        current_action_log_probs = current_log_probs.gather(2, action.long()).squeeze(2)

        # print("current_action_log_probs: ", current_action_log_probs)
        old_probs = batch.action["pi"]
        # old_probs = old_probs[mask == 1]
        old_log_probs = torch.log(old_probs + 1e-16)
        old_action_log_probs = old_log_probs.gather(2, action.long()).squeeze(2)
        # print("old_action_log_probs: ", old_action_log_probs)

        # [seq_len, batch_size]
        ratio = torch.exp(current_action_log_probs - old_action_log_probs) * policy_mask
        # print("ratio: ", ratio)

        # [batch_size]
        reward = batch.reward
        # print("reward: ", reward)
        # reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        advantage = (reward / 7600 - v.squeeze()) * value_mask
        # print("advantage: ", advantage)

        surr1 = ratio * (advantage.detach())
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (advantage.detach())
        # print("surr1: ", surr1)
        # print("surr2: ", surr2)
        # [seq_len, batch_size]
        entropy = -torch.sum(pi * current_log_probs, dim=-1) * policy_mask
        # print("entropy: ", entropy)

        # [seq_len, batch_size]
        policy_loss = -torch.min(surr1, surr2) - entropy_ratio * entropy
        # [seq_len, batch_size]
        value_loss = torch.pow(advantage, 2) * value_loss_weight

        # [seq_len, batch_size]
        priority = advantage.detach()
        # [batch_size]
        priority = priority.abs().sum(0) / batch.seq_len

        return (
            policy_loss.sum(0) / batch.seq_len,
            value_loss.sum(0) / batch.seq_len,
            priority.cpu(),
        )


class BridgeFFWDAgent(torch.jit.ScriptModule):
    __constants__ = ["device", "greedy", "uniform_priority", "reuse_value_in_priority"]

    def __init__(
        self,
        device: str,
        p_in_dim: int,
        v_in_dim: int,
        p_hid_dim: int,
        v_hid_dim: int,
        p_out_dim: int,
        num_p_mlp_layer: int,
        num_v_mlp_layer: int,
        p_activation: str,
        v_activation: str,
        dropout: float,
        net: str,
        greedy: bool = False,
        uniform_priority: bool = False,
        reuse_value_in_priority: bool = False,
    ):
        super().__init__()
        self.device = device
        self.p_in_dim = p_in_dim
        self.v_in_dim = v_in_dim
        self.p_hid_dim = p_hid_dim
        self.v_hid_dim = v_hid_dim
        self.p_out_dim = p_out_dim
        self.num_p_mlp_layer = num_p_mlp_layer
        self.num_v_mlp_layer = num_v_mlp_layer
        self.p_activation = p_activation
        self.v_activation = v_activation
        self.dropout = dropout
        self.net = net
        self.greedy = greedy
        self.uniform_priority = uniform_priority
        self.reuse_value_in_priority = reuse_value_in_priority

        if net == "ws":
            self.network = FFWDA2CWeightSharingNet(
                in_dim=p_in_dim,
                hid_dim=p_hid_dim,
                out_dim=p_out_dim,
                num_mlp_layer=num_p_mlp_layer,
                activation=p_activation,
                dropout=dropout,
            ).to(self.device)
        elif net == "sep":
            self.network = FFWDA2CSeparateNet(
                p_in_dim=p_in_dim,
                v_in_dim=v_in_dim,
                p_hid_dim=p_hid_dim,
                v_hid_dim=v_hid_dim,
                p_out_dim=p_out_dim,
                num_p_mlp_layer=num_p_mlp_layer,
                num_v_mlp_layer=num_v_mlp_layer,
                p_activation=p_activation,
                v_activation=v_activation,
                dropout=dropout,
            ).to(self.device)
        else:
            raise ValueError(f"The net {net} is not supported.")

    def clone(self, device: str, overwrite: Dict = None):
        if overwrite is None:
            overwrite = {}

        cloned = type(self)(
            device,
            self.p_in_dim,
            self.v_in_dim,
            self.p_hid_dim,
            self.v_hid_dim,
            self.p_out_dim,
            self.num_p_mlp_layer,
            self.num_v_mlp_layer,
            self.p_activation,
            self.v_activation,
            overwrite.get("dropout", self.dropout),
            self.net,
            overwrite.get("greedy", self.greedy),
            overwrite.get("uniform_priority", self.uniform_priority),
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.network.forward(obs)

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reply = self.forward(obs)
        assert "legal_pi" in reply
        legal_pi = reply["legal_pi"]
        single_act = legal_pi.dim() == 1
        legal_move = obs["legal_move"]
        if single_act:
            legal_move = legal_move.unsqueeze(0)


        pi_sum = legal_pi.sum(1)
        zero_sel = pi_sum < 1e-5
        # Replace if the sum of pi is too small.
        legal_pi[zero_sel, :] = legal_move[zero_sel, :]
        pi_sum[zero_sel] = legal_pi[zero_sel, :].sum(1)
        legal_pi = legal_pi / pi_sum[:, None]
        reply["legal_pi"] = legal_pi
        greedy_a = torch.argmax(legal_pi, -1).view(-1, 1)

        if self.greedy:
            a = greedy_a
        else:
            a = torch.multinomial(legal_pi, num_samples=1).view(-1, 1)
        reply["a"] = a + 52
        reply["greedy_a"] = greedy_a + 52
        if single_act:
            reply["a"] = reply["a"].squeeze(0)
            reply["greedy_a"] = reply["greedy_a"].squeeze(0)
        return reply

    @torch.jit.script_method
    def compute_priority(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.uniform_priority:
            batch_size = obs["legal_move"].size(0)
            priority = torch.ones(batch_size)
            return {"priority": priority}

        # Use abs(v - r) to compute priority.

        if self.reuse_value_in_priority:
            # Use values computed by old model. This can speed up priority computation.
            assert "v" in obs
            v = obs["v"].squeeze()
        else:
            input_ = {
                "publ_s": obs["publ_s"],
                "priv_s": obs["priv_s"],
                "perf_s": obs["perf_s"],
                "legal_move": obs["legal_move"],
            }
            v = self.forward(input_)["v"].squeeze()  # [batch_size, ]

        r = obs["reward"]  # [batch_size, ]
        adv = v - r
        priority = torch.abs(adv).detach().cpu()

        return {"priority": priority}

    # Python only functions
    def compute_loss_and_priority(
        self,
        batch: pyrela.FFTransition,
        clip_eps: float,
        entropy_ratio: float,
        value_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch.obs
        reply = self.forward(obs)
        v = reply["v"]
        pi = reply["pi"]
        action = batch.action["a"] - 52
        # print(action)
        current_log_probs = torch.log(pi + 1e-16)
        current_action_log_probs = current_log_probs.gather(1, action.long()).squeeze(1)
        old_log_probs = torch.log(batch.action["pi"] + 1e-16)
        old_action_log_probs = old_log_probs.gather(1, action.long()).squeeze(1)

        ratio = torch.exp(current_action_log_probs - old_action_log_probs)
        adv = batch.reward - v.squeeze()
        surr1 = ratio * (adv.detach())
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (adv.detach())

        entropy = -torch.sum(pi * current_log_probs, -1)

        p_loss = -torch.min(surr1, surr2) - entropy_ratio * entropy
        v_loss = torch.pow(adv, 2) * value_weight

        priority = torch.abs(adv).detach().cpu()

        return p_loss, v_loss, priority
