from typing import Dict

import torch
from net import MLP

DEFAULT_POLICY_CONF = {
    "num_fc_layers": 4,
    "in_dim": 480,
    "out_dim": 38,
    "hid_dim": 2048,
    "activation": "gelu",
    "output_activation": "softmax"
}

DEFAULT_VALUE_CONF = {
    "num_fc_layers": 4,
    "in_dim": 480,
    "out_dim": 1,
    "hid_dim": 2048,
    "activation": "gelu",
    "output_activation": ""
}


class BridgeAgent(torch.jit.ScriptModule):

    def __init__(self, policy_conf: Dict, value_conf: Dict):
        super().__init__()
        self.policy_net = MLP.from_conf(policy_conf)
        self.value_net = MLP.from_conf(value_conf)

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        policy = self.policy_net(obs["s"])
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
