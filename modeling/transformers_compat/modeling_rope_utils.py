"""
Compatibility reimplementation of the old transformers.modeling_rope_utils
removed after transformers 4.36.

ThinkMorph expects these symbols:
- ROPE_INIT_FUNCTIONS
- rope_config_validation
- RopeType (optional, depending on code path)
"""

import torch
import math

class RopeType:
    """Minimal stub of the original RopeType enum."""
    NONE = "none"
    DEFAULT = "default"
    LLA = "linear"
    NTK = "ntk"
    DYNAMIC = "dynamic"


def rope_config_validation(config):
    """
    ThinkMorph expects transformers' old validation behavior.
    This stub simply returns the config unchanged.

    Args:
        config: The config object including rope settings.
    Returns:
        config: Unmodified.
    """
    return config


def _dummy_init(config, device=None, **kwargs):
    # dim per head
    dim = config.hidden_size // config.num_attention_heads

    base = getattr(config, "rope_theta", 10000)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Qwen2 models expect a numeric attention scaling, not None
    attention_scaling = getattr(config, "rope_scaling", None)

    if attention_scaling is None:
        attention_scaling = 1.0  

    return inv_freq, attention_scaling

ROPE_INIT_FUNCTIONS = {
    "default": _dummy_init,
    "linear": _dummy_init,
    "ntk": _dummy_init,
    "dynamic": _dummy_init,
}
