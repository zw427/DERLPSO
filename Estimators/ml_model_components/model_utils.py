"""
Utility functions for neural network models.
"""

from typing import List

import numpy as np 
import torch.nn as nn
import torch


def activaton_func(nonlinearity: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        nonlinearity: Name of the activation function
        
    Returns:
        PyTorch activation function module
        
    Raises:
        ValueError: If nonlinearity is not supported
    """
    if nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif nonlinearity.lower() == 'leaky':
        return nn.LeakyReLU()
    elif nonlinearity.lower() == 'tanh':
        return nn.Tanh()
    elif nonlinearity.lower() == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f'Unknown nonlinearity "{nonlinearity}". Accepting relu, leaky, tanh, or elu only.')


def build_modules(hidden_dims: List[int], nonlinearity: str, normalize: bool = False):
    """
    Build a sequential neural network with specified dimensions and activation.
    
    Args:
        hidden_dims: List/tensor of layer dimensions [input_dim, hidden1, ..., output_dim]
        nonlinearity: Name of activation function to use
        normalize: Whether to add sigmoid normalization at the end
        
    Returns:
        nn.Sequential: Neural network module
    """
    if len(hidden_dims) < 2:
        raise ValueError("hidden_dims must contain at least input and output dimensions")
    
    modules = []
    
    # Build all layers except the last one 
    for i in range(len(hidden_dims) - 1):
        input_dim = hidden_dims[i]
        output_dim = hidden_dims[i + 1]
        
        # Add linear layer
        linear_layer = nn.Linear(input_dim, output_dim, bias=False)
        
        if i < len(hidden_dims) - 2:  # Not the output layer
            # Add activation for hidden layers
            modules.append(nn.Sequential(
                linear_layer,
                activaton_func(nonlinearity)
            ))
        else:  # Output layer (no activation)
            modules.append(linear_layer)
    
    # Add normalization if requested
    if normalize:
        modules.append(nn.Sigmoid())
    
    return nn.Sequential(*modules)


def create_net(input_dim: int, output_dim: int, n_layers: int = 1,
               hidden_dim: int = 100, nonlinear: nn.Module = nn.Tanh, dropout: bool = False):
    """
    Create a fully connected neural network.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        n_layers: Number of hidden layers
        hidden_dim: Number of units per hidden layer
        nonlinear: Activation function class
        dropout: Whether to add dropout layers
        
    Returns:
        nn.Sequential: Neural network
    """
    layers = [nn.Linear(input_dim, hidden_dim)]
    
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        if dropout and i < n_layers - 1:  # Add dropout except for last layer
            layers.append(nn.Dropout(0.2))
    
    layers.append(nonlinear())
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


def init_network_weights(net, std: float = 0.1):
    """
    Initialize network weights with normal distribution.
    
    Args:
        net: Single network or list of networks to initialize
        std: Standard deviation for normal distribution
    """
    networks = [net] if not isinstance(net, list) else net
    
    for network in networks:
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, val=0.0)


def linspace_vector(start: torch.Tensor, end: torch.Tensor, n_points: int):
    """
    Create linear interpolation between start and end tensors.
    
    Args:
        start: Starting tensor (scalar or vector)
        end: Ending tensor (must have same shape as start)
        n_points: Number of interpolation points
        
    Returns:
        Interpolated tensor
    """
    assert start.size() == end.size(), "Start and end tensors must have the same size"
    
    size = np.prod(start.size())
    
    if size == 1:
        # Handle scalar case
        return torch.linspace(start, end, n_points, dtype=torch.float64)
    else:
        # Handle vector case
        result = torch.Tensor()
        for i in range(start.size(0)):
            interpolated = torch.linspace(start[i], end[i], n_points, dtype=torch.float64)
            result = torch.cat((result, interpolated), 0)
        return torch.t(result.reshape(start.size(0), n_points))


def sample_standard_gaussian(mu: torch.Tensor, sigma: torch.Tensor):
    """
    Sample from Gaussian distribution using reparameterization trick.
    
    Args:
        mu: Mean tensor
        sigma: Standard deviation tensor
        
    Returns:
        Sampled tensor with same shape as mu and sigma
    """
    device = mu.device
    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma + mu


def split_last_dim(data: torch.Tensor, split_evenly: bool = True, split_pos: int = None):
    """
    Split tensor along the last dimension.
    
    Args:
        data: Input tensor to split
        split_evenly: If True, split evenly in half. If False, use split_pos
        split_pos: Position to split at (only used if split_evenly=False)
        
    Returns:
        Tuple of (first_part, second_part)
    """
    last_dim = data.size(-1)
    
    if split_evenly:
        if last_dim % 2 != 0:
            raise ValueError("Cannot split evenly: last dimension must be even")
        split_point = last_dim // 2
    else:
        if split_pos is None:
            raise ValueError("split_pos must be provided when split_evenly=False")
        split_point = split_pos
    
    # Use ellipsis for cleaner tensor slicing
    first_part = data[..., :split_point]
    second_part = data[..., split_point:]
    
    return first_part, second_part
