"""
Compatibility stub for transformers.modeling_flash_attention_utils
removed in Transformers >= 4.37.

ThinkMorph expects:
- _flash_attention_forward

This implementation falls back to a standard attention forward pass.
"""

import torch
import torch.nn.functional as F


def _flash_attention_forward(
    query,
    key,
    value,
    attention_mask=None,
    dropout=0.0,
    softmax_scale=None,
    training=False,
):
    """
    A simple safe fallback implementation of attention for environments
    without FlashAttention kernels.
    """

    scores = torch.matmul(query, key.transpose(-2, -1))

    if softmax_scale is not None:
        scores = scores * softmax_scale
    else:
        scores = scores / (query.size(-1) ** 0.5)

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_probs = F.softmax(scores, dim=-1)

    if dropout > 0 and training:
        attn_probs = F.dropout(attn_probs, p=dropout)

    output = torch.matmul(attn_probs, value)

    return output
