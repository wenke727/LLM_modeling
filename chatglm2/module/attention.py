import math
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from loguru import logger


from ..configuration_chatglm import ChatGLMConfig
from .position_embedding import apply_rotary_pos_emb
from .misc import _config_to_kwargs, split_tensor_along_last_dim


class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number: int, verbose_level='trace'):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling # True
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32 # True
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size # 4096
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads # 128 = 4096 // 32
        self.num_attention_heads_per_partition = config.num_attention_heads # 32

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)
        self.verbose_level = verbose_level

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        pytorch_major_version = int(torch.__version__.split('.')[0])
    
        if pytorch_major_version >= 2:
            """scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)"""
            # [seq_len, bs, 32, 128] --> [bs, 32, seq_len, 128]
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

            # verbose
            if self.layer_number == 28:
                getattr(logger, self.verbose_level)(
                    f"Q: {list(query_layer.shape)}, KV: {list(key_layer.shape)}, "
                    f"causal: {attention_mask is None and query_layer.shape[2] == key_layer.shape[2]}, "
                    f"attention_mask: {attention_mask}")
            
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                # step 1: Causual Attention
                context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, is_causal=True)
            else:
                # step 2 ~ n: `The sequence length of Query is 1`, Q: [bs, 32, 1, 128], K, V: [bs, 32, seq++, 128]
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask) # None (eval mode)
            
            # Reshape: [bs, 32, seq_len, 128] --> [seq_len, bs, 32, 128] --> [seq_len, bs, 4096]
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # TODO: Raw attention scores

            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] --> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] --> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer --> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None, verbose_level='debug'):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads # 4096 = 128 * 32

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads # 128 = 4096 // 32
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention # True
        self.qkv_hidden_size = 3 * self.projection_size # 12288
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num # Group Query Attention: 2
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            ) # 4608 = (32 + 2 + 2) * num_heads
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size, # 4096, 4608
                                         bias=config.add_bias_linear or config.add_qkv_bias, # True
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================

        # Step 1. Attention heads [sq, b, h] --> [sq, b, (32 + 2 + 2) * 128]
        mixed_x_layer = self.query_key_value(hidden_states)

        # Step 2. Split to `QKV`
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head, # 32, 128
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head, # 2, 128
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head, # 2, 128
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            ) # [seq_len, bs, 4096] --> [seq_len, bs, `32`, 128]
            
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            ) # [seq_len, bs, 256] --> [seq_len, bs, `2`, 128]
            value_layer = value_layer.view(
                value_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # Step 3. apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # Step 4. kv_cache: adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0) # [seq + 1, bs, 32, 128] <-- [seq, ...] + [1, ...]
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer) if use_cache else None
        else:
            kv_cache = None

        # Step 5. Repeat `key_layer` and `value_layer`
        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2) # [seq_len, bs, 2, 128] --> [seq_len, bs, 2, `1`, 128]
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            ) # [seq_len, bs, 2, 1, 128] --> # [seq_len, bs, 2, `16`, 128]
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            ) # [seq_len, bs, 2, 16, 128] --> [seq_len, bs, `32`, 128]
            
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # Step 6. core attention computation, 1) query: [seq, 1, 1, ..., 1]
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # Step 7. Output. [sq, b, h]
        output = self.dense(context_layer)

        return output, kv_cache

