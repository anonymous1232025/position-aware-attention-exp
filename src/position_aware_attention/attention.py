"""
Position-Aware Attention Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .position_effect import PositionEffectFunction, EnhancedPositionEffectFunction


class PositionAwareAttentionLayer(nn.Module):
    """
    Position-Aware Attention Layer
    
    Implements attention mechanism with explicit position-attention relationship.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 12,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: Optional[float] = None,
        dropout: float = 0.1,
        use_enhanced: bool = True
    ):
        """
        Initialize position-aware attention layer.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            alpha: Position influence strength parameter
            beta: Position decay parameter
            gamma: Position enhancement parameter (for enhanced function)
            dropout: Dropout rate
            use_enhanced: Whether to use enhanced position effect function
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Initialize position effect function
        if use_enhanced and gamma is not None:
            self.position_effect = EnhancedPositionEffectFunction(alpha, beta, gamma)
        else:
            self.position_effect = PositionEffectFunction(alpha, beta)
        
        # Linear projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Attention mask [batch_size, seq_len, seq_len] or [batch_size, seq_len]
            position_matrix: Pre-computed position effect matrix [seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.query_proj(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply position effect matrix
        if position_matrix is None:
            position_matrix = self.position_effect.get_position_matrix(seq_len)
        
        # Move position_matrix to same device as scores
        position_matrix = position_matrix.to(scores.device)
        
        # Expand position matrix for batch and heads
        position_matrix = position_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Combine attention scores with position effects
        scores = scores + torch.log(position_matrix + 1e-9)  # Log-space combination
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output)
        
        return output, attention_weights.mean(dim=1)  # Average over heads for visualization

