
"""
Encoder-Decoder models for time series processing.

"""
from typing import List

import torch 
import torch.nn as nn

from models.ml_model_components.gated_recurrent_unit import GRUUnitOriginal, GRUUnit
from models.ml_model_components.model_utils import linspace_vector, build_modules


class RNN(nn.Module):
    """Simple RNN encoder for time series data."""
    
    def __init__(self, latent_dim: int, input_dim: int, GRU_update=None, 
                 n_GRUUnits: int = 20, device: torch.device = torch.device("cpu")):
        super(RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRUUnitOriginal(latent_dim, input_dim, n_units=n_GRUUnits).to(device)
        else:
            self.GRU_update = GRU_update

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

    def forward(self, data: torch.Tensor, time_steps: torch.Tensor):
        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()
        return self.run_odernn(data, time_steps)

    def run_odernn(self, data: torch.Tensor, time_steps: torch.Tensor):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        n_traj = data.shape[0]

        # Initialize latent state
        prev_y = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(self.device)

        latent_ys = []
        for i in range(len(time_steps)):
            xi = data[:, i, :]
            yi = self.GRU_update(prev_y, xi)
            prev_y = yi
            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, dim=1)

        assert not torch.isnan(yi).any()
        return yi, latent_ys


class ODE_RNN(nn.Module):
    """ODE-RNN encoder combining neural ODEs with recurrent units."""
    
    def __init__(self, latent_dim: int, input_dim: int, z0_diffeq_solver=None, 
                 GRU_update=None, n_GRUUnits: int = 100, device: torch.device = torch.device("cpu")):
        super(ODE_RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRUUnitOriginal(latent_dim, input_dim, n_units=n_GRUUnits).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

    def forward(self, data: torch.Tensor, time_steps: torch.Tensor):
        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()
        return self.run_odernn(data, time_steps)

    def run_odernn(self, data: torch.Tensor, time_steps: torch.Tensor):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        n_traj = data.shape[0]

        # Initialize latent state
        prev_y = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(self.device)

        prev_t, t_i = time_steps[0], time_steps[1]
        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        latent_ys = []

        for i in range(1, len(time_steps)):
            if abs(prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i)).to(self.device)
                # dt*f'
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert not torch.isnan(inc).any()

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), dim=2).to(self.device)

                assert not torch.isnan(ode_sol).any()
            else:
                n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp).to(self.device)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert not torch.isnan(ode_sol).any()

            if torch.mean(ode_sol[:, :, 0] - prev_y) >= 0.001:
                Exception(f"Error: first point of the ODE is not equal to initial value {torch.mean(ode_sol[:, 0, :] - prev_y)}")

            yi_ode = ode_sol[:, :, -1]
            xi = data[:, i, :]
            yi = self.GRU_update(yi_ode, xi)

            prev_y = yi
            # ERROR: assignments of prev_t and t_i might be wrong
            prev_t, t_i = time_steps[i], time_steps[i - 1]
            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, dim=1)

        assert not torch.isnan(yi).any()
        return yi, latent_ys


class VAE_ODE_RNN(nn.Module):
    """Variational Autoencoder with ODE-RNN encoder."""
    
    def __init__(self, latent_dim: int, input_dim: int, z0_diffeq_solver=None, 
                 GRU_update=None, n_GRUUnits: int = 100, device: torch.device = torch.device("cpu")):
        super(VAE_ODE_RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRUUnit(latent_dim, input_dim, n_units=n_GRUUnits).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

    def forward(self, data: torch.Tensor, time_steps: torch.Tensor):
        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()
        return self.run_odernn(data, time_steps)

    def run_odernn(self, data: torch.Tensor, time_steps: torch.Tensor):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        n_traj = data.shape[0]

        # Initialize latent state and variance
        prev_y = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(self.device)
        prev_logvar = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(self.device)

        prev_t, t_i = time_steps[0], time_steps[1]
        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        latent_ys = []
        
        for i in range(1, len(time_steps)):
            if abs(prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i)).to(self.device)
                # dt*f'
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert not torch.isnan(inc).any()

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), dim=2).to(self.device)

                assert not torch.isnan(ode_sol).any()
            else:
                n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp).to(self.device)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert not torch.isnan(ode_sol).any()

            if torch.mean(ode_sol[:, :, 0] - prev_y) >= 0.001:
                Exception(f"Error: first point of the ODE is not equal to initial value {torch.mean(ode_sol[:, 0, :] - prev_y)}")

            yi_ode = ode_sol[:, :, -1]
            xi = data[:, i, :]
            yi, yi_logvar = self.GRU_update(yi_ode, prev_logvar, xi)

            prev_y, prev_logvar = yi, yi_logvar
            # ERROR: assignments of prev_t and t_i might be wrong
            prev_t, t_i = time_steps[i], time_steps[i - 1]

            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, dim=1)

        assert not torch.isnan(yi).any()
        assert not torch.isnan(yi_logvar).any()
        return yi, yi_logvar
    

class VAE_RNN(nn.Module):
    """Variational Autoencoder with RNN encoder."""
    
    def __init__(self, latent_dim: int, input_dim: int, GRU_update=None, 
                 n_GRUUnits: int = 20, device: torch.device = torch.device("cpu")):
        super(VAE_RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRUUnit(latent_dim, input_dim, n_units=n_GRUUnits).to(device)
        else:
            self.GRU_update = GRU_update

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

    def forward(self, data: torch.Tensor, time_steps: torch.Tensor):
        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()
        return self.run_odernn(data, time_steps)

    def run_odernn(self, data: torch.Tensor, time_steps: torch.Tensor):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        n_traj = data.shape[0]

        # Initialize latent state and variance
        prev_y = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(self.device)
        prev_logvar = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(self.device)

        for i in range(len(time_steps)):
            xi = data[:, i, :]
            yi, logvar = self.GRU_update(prev_y, prev_logvar, xi)
            prev_y = yi
            prev_logvar = logvar

        assert not torch.isnan(yi).any()
        assert not torch.isnan(logvar).any()
        return yi, logvar
    

class MLP(nn.Module):
    """Multi-Layer Perceptron for time series processing."""
    
    def __init__(self, hidden_dims: List[int], nonlinearity: str):
        super(MLP, self).__init__()
        self.model = build_modules(hidden_dims, nonlinearity)

    def forward(self, data: torch.Tensor, time_steps: torch.Tensor):
        data = data.flatten(start_dim=1, end_dim=len(data.shape) - 1)
        return self.model(data)


class DecoderMLP(nn.Module):
    """MLP-based decoder wrapper."""
    
    def __init__(self, model: nn.Module):
        super(DecoderMLP, self).__init__()
        self.model = model

    def forward(self, data: torch.Tensor, time_steps: torch.Tensor):
        return self.model(data)


class DecoderODE_RNN(nn.Module):
    """ODE-RNN decoder for generating time series from latent representations."""
    
    def __init__(self, input_dim_param: int, dim_of_data: int, latent_dim: int, 
                 diffeq_solver: nn.Module, GRU_update=None, n_GRUUnits: int = 100, 
                 device: torch.device = torch.device("cpu")):
        super(DecoderODE_RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRUUnitOriginal(input_dim_param, dim_of_data, n_units=n_GRUUnits).to(device)
        else:
            self.GRU_update = GRU_update

        self.dim_of_data = dim_of_data
        self.device = device
        self.diffeq_solver = diffeq_solver
        self.latent_dim = latent_dim

    def forward(self, data: torch.Tensor, time_steps: torch.Tensor, prior: torch.Tensor):
        assert not torch.isnan(data).any()
        assert not torch.isnan(time_steps).any()
        return self.run_odernn(data, time_steps, prior)

    def run_odernn(self, data: torch.Tensor, time_steps: torch.Tensor, prior: torch.Tensor = None):
        """Decode latent representation through ODE-RNN."""
        n_traj = data.shape[0]

        # Initialize with prior or zeros
        prev_y = prior
        if prior is None:
            prev_y = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(self.device)

        prev_t, t_i = time_steps[0], time_steps[1]
        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        for i in range(1, len(time_steps)):
            if abs(prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i)).to(self.device)
                # dt*f'
                inc = self.diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert not torch.isnan(inc).any()

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), dim=2).to(self.device)

                assert not torch.isnan(ode_sol).any()
            else:
                n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp).to(self.device)
                ode_sol = self.diffeq_solver(prev_y, time_points)

                assert not torch.isnan(ode_sol).any()

            if torch.mean(ode_sol[:, :, 0] - prev_y) >= 0.001:
                Exception(f"Error: first point of the ODE is not equal to initial value {torch.mean(ode_sol[:, 0, :] - prev_y)}")

            yi_ode = ode_sol[:, :, -1]
            xi = data[:, i, :]
            yi = self.GRU_update(yi_ode, xi)
            prev_y = yi
            # ERROR: assignments of prev_t and t_i might be wrong
            prev_t, t_i = time_steps[i], time_steps[i - 1]

        return prev_y
