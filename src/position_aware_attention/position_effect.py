"""
Position Effect Function Implementation
Implements the mathematical framework for position-aware attention.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class PositionEffectFunction:
    """
    Position Effect Function Class
    
    Implements the basic position effect function:
    P_effect(i, j, L) = α * e^(-β * |i-j|/L)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 2.0):
        """
        Initialize position effect function.
        
        Args:
            alpha: Position influence strength parameter
            beta: Position decay parameter
        """
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, i: int, j: int, L: int) -> float:
        """
        Calculate position effect function.
        
        Args:
            i: Query position
            j: Key position
            L: Sequence length
            
        Returns:
            float: Position influence weight
        """
        distance = abs(i - j)
        normalized_distance = distance / L if L > 0 else 0
        
        return self.alpha * math.exp(-self.beta * normalized_distance)
    
    def get_position_matrix(self, L: int) -> torch.Tensor:
        """Get complete position effect matrix for sequence length L."""
        matrix = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                matrix[i, j] = self(i, j, L)
        return matrix


class EnhancedPositionEffectFunction:
    """
    Enhanced Position Effect Function Class
    
    Implements the enhanced position effect function with gamma parameter:
    P_effect(i, j, L) = α * (1 + γ * e^(-β * |i-j|/L)) / (1 + γ)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.5):
        """
        Initialize enhanced position effect function.
        
        Args:
            alpha: Position influence strength parameter
            beta: Position decay parameter
            gamma: Position enhancement parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, i: int, j: int, L: int) -> float:
        """
        Calculate enhanced position effect function.
        
        Args:
            i: Query position
            j: Key position
            L: Sequence length
            
        Returns:
            float: Enhanced position influence weight
        """
        distance = abs(i - j)
        normalized_distance = distance / L if L > 0 else 0
        
        base_effect = math.exp(-self.beta * normalized_distance)
        enhanced_effect = (1 + self.gamma * base_effect) / (1 + self.gamma)
        
        return self.alpha * enhanced_effect
    
    def get_position_matrix(self, L: int) -> torch.Tensor:
        """Get complete enhanced position effect matrix for sequence length L."""
        matrix = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                matrix[i, j] = self(i, j, L)
        return matrix
    
    def get_optimal_position(self, L: int) -> int:
        """
        Calculate optimal position for information placement.
        
        Args:
            L: Sequence length
            
        Returns:
            int: Optimal position index
        """
        # Optimal position is at the center of the sequence
        return L // 2

