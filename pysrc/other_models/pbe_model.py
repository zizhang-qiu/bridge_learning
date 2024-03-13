from typing import Dict, List
import torch
import torch.jit
from torch import nn
import scipy
import scipy.io as sio


class PBEModel(torch.jit.ScriptModule):
    __constants__ = ["num_action", "hidden_dim", "input_dim", "hand_dim", "bid_index"]

    def __init__(self):
        """Model proposed in ""Automatic Bridge Bidding Using Deep Reinforcement Learning,
        copied from JPS repo"".

        Raises:
            RuntimeError: If the model file cannot be found.
        """
        super().__init__()

        mat_content = sio.loadmat(
            "../external_models/pbe/model_valcost_1.063088e-01_totalbid4_4_128_50_3.204176e-03_9.800000e-01_8.200000e-01_alpha1.000000e-01.mat"
        )
        if mat_content is None:
            raise RuntimeError("Cannot find model!")

        # No double and redouble.
        self.num_action = 36
        self.hidden_dim = 128
        self.hand_dim = 52
        self.bid_index = 93
        self.input_dim = 94

        self.linears = torch.nn.ModuleList([nn.Linear(1, 1) for i in range(20)])
        self.leaky_relu = nn.LeakyReLU(0.2)

        for i in range(4):
            for j in range(5):
                if j == 0:
                    if i == 0:
                        in_channel = self.hand_dim
                    else:
                        in_channel = self.bid_index
                else:
                    in_channel = self.hidden_dim
                if j == 4:
                    out_channel = self.num_action
                else:
                    out_channel = self.hidden_dim

                self.linears[i * 5 + j] = torch.nn.Linear(in_channel, out_channel)
                self.linears[i * 5 + j].weight = torch.nn.Parameter(
                    torch.from_numpy(mat_content["WW_qlearning"][0][i][0][j]).cuda()
                )
                self.linears[i * 5 + j].bias = torch.nn.Parameter(
                    torch.from_numpy(mat_content["BB_qlearning"][0][i][0][j])
                    .squeeze()
                    .cuda()
                )

    @torch.jit.script_method
    def act(self, obs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # We change the key to "pbe_s".
        x = obs["pbe_s"].cuda()

        bs = x.size(0)
        # assert((x[:, self.bid_index] < 4).all())
        result = torch.zeros(bs, self.num_action).cuda()
        count = 0
        xi = x
        mask = x[:, self.bid_index] == 0
        for linear in self.linears:
            i = count // 5
            j = count % 5
            if j == 0:
                mask = x[:, self.bid_index] == i
                if i == 0:
                    xi = x.narrow(1, 0, self.hand_dim)
                else:
                    xi = x.narrow(1, 0, self.bid_index)
            xi = linear(xi)
            xi = self.leaky_relu(xi)
            if j == 4:
                result += xi * mask.float().unsqueeze(1).expand_as(xi)
            count += 1

        return {"pi": result.detach().cpu()}

    @torch.jit.script_method
    def act_greedy(self, obs:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reply = self.act(obs)
        pi = reply["pi"]

        batch_size = pi.size(0)
        a = torch.zeros(batch_size, dtype=torch.int32)
        for i in range(batch_size):
            min_cost = 1
            best_a = 52
            for j in range(pi[i].size(0) - 1, 0, -1):
                if obs["jps_legal_move"][i][j-1].item() == 1.0:
                    if pi[i][j] < min_cost:
                        min_cost = pi[i][j]
                        best_a = j - 1 + 3 + 52
                else:
                    break
            
            if min_cost <= 0.2:
                a[i] = best_a
            else:
                a[i] = 52
            
            if pi[i][0] < min_cost:
                a[i] = 52
                
        return {"a": a}
