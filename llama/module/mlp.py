import torch
from torch import nn
from loguru import logger
import torch.nn.functional as F
from transformers.activations import ACT2FN


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # ? why 13824: 5120 * 4 / 3 * 2 = 13653,  108 * 128 = 13824 
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)    # 5120, 13824
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)      # 5120, 13824
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)    # 5120, 13824
        self.act_fn = ACT2FN[config.hidden_act] # silu

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # ((bs, len, 13824) * (bs, len, 13824))
            # $Swish_{\beta}(x, W, V, b, c, \beta) = Swish_{\beta}(xW + b) \otimes (xV + c)$, normaly $\beta = 1$
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

