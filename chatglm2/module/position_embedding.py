import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim # 4096 / 32 / 2 
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)) # [32]

        # Create position indexes `[0, 1, ..., seq_len - 1]`, seq_len = 32768
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float() # [seq_len, 32]

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1) # [seq_len, 32, 2]

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # rope_cache: [[sq, b, 32, 2]], `ChatGLMModel.forward` 已做 slice
    # x: [sq, b, np, hn], np 32, hn 128
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2 # 64
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2) # [sq, bs, 32, 64] -> [sq, bs, 32, 32, 2]
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2) # [sq, bs, 32, 2] -> [sq, bs, 1, 32, 2]
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    ) # [(sq, bs, 32, 32), (sq, bs, 32, 32)] -> (sq, bs, 32, 32, 2)
    x_out2 = x_out2.flatten(3) # [sq, bs, 32, 32, 2] -> [sq, bs, 32, 64]
    
    return torch.cat((x_out2, x_pass), dim=-1) # [sq, bs, 32, 64] -> [sq, bs, 32, 128]

