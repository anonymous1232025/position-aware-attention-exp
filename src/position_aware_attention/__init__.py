"""
Position-Aware Attention Mechanism
A mathematical framework for enhanced spatial information processing in Transformer architectures.
"""

from .models import (
    PositionAttentionConfig,
    PositionAwareAttention,
    TripleAttentionArchitecture
)
from .attention import PositionAwareAttentionLayer
from .position_effect import PositionEffectFunction, EnhancedPositionEffectFunction
from .triple_attention import TripleAttention

__version__ = "1.0.0"
__all__ = [
    "PositionAttentionConfig",
    "PositionAwareAttention",
    "TripleAttentionArchitecture",
    "PositionAwareAttentionLayer",
    "PositionEffectFunction",
    "EnhancedPositionEffectFunction",
    "TripleAttention",
]

