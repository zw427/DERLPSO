"""
Base RNN components for neural ODE models.

This module contains GRU-based units used in neural ODE architectures,
including original GRU units and variational GRU units for handling
uncertainty in latent representations.
"""
import torch
import torch.nn as nn
from typing import Tuple

from models.ml_model_components.model_utils import init_network_weights, split_last_dim


class GRUUnitOriginal(nn.Module):
    """
    Original GRU unit implementation for neural ODE models.
    
    This is a standard GRU cell that processes latent state and input
    to produce updated latent state.
    """
    
    def __init__(self, latent_dim: int, input_dim: int, n_units: int = 20):
        """
        Initialize the original GRU unit.
        
        Args:
            latent_dim: Dimension of the latent state
            input_dim: Dimension of the input
            n_units: Number of hidden units in the gates
            device: Device to place the module on
        """
        super(GRUUnitOriginal, self).__init__()
        
        # Update gate: controls how much of the previous state to keep
        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        init_network_weights(self.update_gate)

        # Reset gate: controls how much of the previous state to forget
        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        init_network_weights(self.reset_gate)

        # New state network: generates candidate new state
        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Linear(n_units, latent_dim),
            nn.Tanh()
        )
        init_network_weights(self.new_state_net)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GRU unit.
        
        Args:
            y: Previous latent state [batch_size, latent_dim]
            x: Current input [batch_size, input_dim]
            
        Returns:
            Updated latent state [batch_size, latent_dim]
        """
        y_concat = torch.cat([y, x], dim=-1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        
        # Apply reset gate to previous state before computing new state
        concat = torch.cat([y * reset_gate, x], dim=-1)
        new_state = self.new_state_net(concat)
        
        # Combine previous state and new state using update gate
        new_y = (1 - update_gate) * new_state + update_gate * y

        assert not torch.isnan(new_y).any(), "NaN detected in GRU output"
        return new_y


class GRUUnit(nn.Module):
    """
    Variational GRU unit for handling uncertainty in latent representations.
    
    This GRU unit processes both mean and standard deviation of latent states,
    enabling variational inference in neural ODE models.
    """
    
    def __init__(self, latent_dim: int, input_dim: int, n_units: int = 100):
        """
        Initialize the variational GRU unit.
        
        Args:
            latent_dim: Dimension of the latent state
            input_dim: Dimension of the input
            n_units: Number of hidden units in the gates
        """
        super(GRUUnit, self).__init__()

        # Update gate: controls how much of the previous state to keep
        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        init_network_weights(self.update_gate)

        # Reset gate: controls how much of the previous state to forget
        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        init_network_weights(self.reset_gate)

        # New state network: generates candidate new state (mean and std)
        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2)
        )
        init_network_weights(self.new_state_net)

    def forward(self, y_mean: torch.Tensor, y_std: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the variational GRU unit.
        
        Args:
            y_mean: Previous latent state mean [batch_size, latent_dim]
            y_std: Previous latent state std [batch_size, latent_dim]
            x: Current input [batch_size, input_dim]
            
        Returns:
            Tuple of (updated latent mean, updated latent std)
        """
        y_concat = torch.cat([y_mean, y_std, x], dim=-1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        
        # Apply reset gate to previous state before computing new state
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], dim=-1)
        
        # Split the output into mean and std components
        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()  # Ensure positive std

        # Combine previous and new states using update gate
        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert not torch.isnan(new_y).any(), "NaN detected in GRU mean output"
        
        new_y_std = new_y_std.abs()  # Ensure positive std
        return new_y, new_y_std
