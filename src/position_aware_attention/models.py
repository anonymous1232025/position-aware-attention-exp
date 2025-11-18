"""
Model Architectures
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from .attention import PositionAwareAttentionLayer
from .position_effect import EnhancedPositionEffectFunction


@dataclass
class PositionAttentionConfig:
    """Configuration for Position-Aware Attention models."""
    sequence_length: int = 512
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 1.5
    dropout: float = 0.1
    use_enhanced: bool = True


class PositionAwareAttention(nn.Module):
    """
    Position-Aware Attention Model
    
    A Transformer-based model with explicit position-attention relationship.
    """
    
    def __init__(self, config: PositionAttentionConfig):
        super().__init__()
        self.config = config
        
        # Position effect function
        if config.use_enhanced:
            self.position_effect = EnhancedPositionEffectFunction(
                config.alpha, config.beta, config.gamma
            )
        else:
            from .position_effect import PositionEffectFunction
            self.position_effect = PositionEffectFunction(config.alpha, config.beta)
        
        # Multi-layer attention
        self.layers = nn.ModuleList([
            PositionAwareAttentionLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma if config.use_enhanced else None,
                dropout=config.dropout,
                use_enhanced=config.use_enhanced
            )
            for _ in range(config.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights_list)
        """
        # Add position embedding
        seq_len = x.size(1)
        x = x + self.position_embedding[:, :seq_len, :]
        
        attention_weights_list = []
        
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            # Pre-norm architecture
            x_norm = layer_norm(x)
            x_attn, attn_weights = layer(x_norm, mask)
            x = x + x_attn
            attention_weights_list.append(attn_weights)
        
        return x, attention_weights_list


class TripleAttentionArchitecture(nn.Module):
    """
    Triple-Attention Architecture
    
    Combines three attention mechanisms:
    1. Task-aware attention
    2. Content-aware attention
    3. Position-aware attention
    """
    
    def __init__(self, config: PositionAttentionConfig):
        super().__init__()
        self.config = config
        
        # Three attention mechanisms
        self.task_aware_attention = PositionAwareAttentionLayer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma if config.use_enhanced else None,
            dropout=config.dropout,
            use_enhanced=config.use_enhanced
        )
        
        self.content_aware_attention = PositionAwareAttentionLayer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma if config.use_enhanced else None,
            dropout=config.dropout,
            use_enhanced=config.use_enhanced
        )
        
        self.position_aware_attention = PositionAwareAttentionLayer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma if config.use_enhanced else None,
            dropout=config.dropout,
            use_enhanced=config.use_enhanced
        )
        
        # Fusion layer
        self.fusion = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Attention mask
            task_embedding: Task embedding for task-aware attention
            
        Returns:
            Tuple of (output, attention_dict)
        """
        # Task-aware attention
        task_out, task_attn = self.task_aware_attention(
            x if task_embedding is None else x + task_embedding,
            mask
        )
        
        # Content-aware attention
        content_out, content_attn = self.content_aware_attention(x, mask)
        
        # Position-aware attention
        position_out, position_attn = self.position_aware_attention(x, mask)
        
        # Fusion
        fused = torch.cat([task_out, content_out, position_out], dim=-1)
        output = self.fusion(fused)
        output = self.layer_norm(output)
        
        attention_dict = {
            'task_aware': task_attn,
            'content_aware': content_attn,
            'position_aware': position_attn
        }
        
        return output, attention_dict

