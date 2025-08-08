"""
Base parameter estimation models for distribution estimation.

This module contains baseline models for parameter estimation including
encoder-decoder architectures and variational autoencoders.
"""

from typing import Dict, Tuple, Any

from Estimators.ml_model_components.model_utils import sample_standard_gaussian

import torch
import torch.nn as nn


class Baseline(nn.Module):
    """Base class for parameter estimation models."""
    
    def __init__(self):
        super(Baseline, self).__init__()

    def compute(self, data: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted parameters from input data.
        
        Args:
            data: Input time series data
            time: Time steps corresponding to the data
            
        Returns:
            Predicted parameters
        """
        info = self.get_reconstruction(data, time)
        pred_param = info["pred"]
        return pred_param

    def get_reconstruction(self, truth: torch.Tensor, truth_time_steps: torch.Tensor) -> Dict[str, Any]:
        """
        Get reconstruction information. Should be implemented by subclasses.
        
        Args:
            truth: Ground truth data
            truth_time_steps: Time steps for the truth data
            
        Returns:
            Dictionary containing reconstruction information
        """
        raise NotImplementedError("Subclasses must implement get_reconstruction method")


class EncoderDecoder(Baseline):
    """Encoder-Decoder model for parameter estimation."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Initialize encoder-decoder model.
        
        Args:
            encoder: Encoder network
            decoder: Decoder network
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def get_reconstruction(self, truth: torch.Tensor, truth_time_steps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get reconstruction using encoder-decoder architecture.
        
        Args:
            truth: Ground truth data
            truth_time_steps: Time steps for the truth data
            
        Returns:
            Dictionary containing predicted parameters
        """
        h_mu, _ = self.encoder(truth, truth_time_steps)
        
        # Concatenate flattened truth with encoded representation
        truth_flat = torch.cat((truth.flatten(1, 2), h_mu), dim=-1)
        pred_param = self.decoder(truth_flat, truth_time_steps)
        
        return {"pred": pred_param}


class VAE(Baseline):
    """Variational Autoencoder for parameter estimation."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, transform: nn.Moduloe):
        """
        Initialize VAE model.
        
        Args:
            encoder: Encoder network that outputs mean and std
            transform: Transformation network for latent space
            decoder: Decoder network
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transform = transform


    def compute(self, data: torch.Tensor, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE outputs including parameters, mean, and standard deviation.
        
        Args:
            data: Input time series data
            time: Time steps corresponding to the data
            
        Returns:
            Tuple of (predicted parameters, mean, standard deviation)
        """
        info = self.get_reconstruction(data, time)
        pred_param = info["pred"]
        mu = info["mu"]
        std = info["std"]
        return pred_param, mu, std

    def get_reconstruction(self, truth: torch.Tensor, truth_time_steps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get VAE reconstruction with latent variable sampling.
        
        Args:
            truth: Ground truth data
            truth_time_steps: Time steps for the truth data
            
        Returns:
            Dictionary containing predicted parameters, mean, and std
        """
        h_mu, h_std = self.encoder(truth, truth_time_steps)
        
        # Sample from the latent distribution
        z = sample_standard_gaussian(h_mu, h_std)
        
        # Transform latent variable and decode
        prior = self.transform(z)
        pred_param = self.decoder(truth, truth_time_steps, prior)
        
        return {
            "pred": pred_param,
            "mu": h_mu,
            "std": h_std
        }


class Base(Baseline):
    """Simple base model wrapper."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize base model.
        
        Args:
            model: The underlying model to wrap
        """
        super(Base, self).__init__()
        self.model = model

    def get_reconstruction(self, truth: torch.Tensor, truth_time_steps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get reconstruction using the wrapped model.
        
        Args:
            truth: Ground truth data
            truth_time_steps: Time steps for the truth data
            
        Returns:
            Dictionary containing predicted parameters
        """
        pred_param = self.model(truth, truth_time_steps)
        return {"pred": pred_param}
    