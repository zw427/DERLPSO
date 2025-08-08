"""
Model factory for creating various neural network architectures.
"""
import torch 
import torch.nn as nn

from Estimators.ml_model_components.diffeq_solver import DiffeqSolver
from Estimators.ml_model_components.encoder_decoder import VAE_ODE_RNN, VAE_RNN, RNN, ODE_RNN, MLP, DecoderMLP, DecoderODE_RNN
from Estimators.ml_model_components.model_utils import create_net, build_modules
from Estimators.ml_model_components.ode_func import ODEFunc
from Estimators.ml_model_components.parameter_estimators import VAE, EncoderDecoder, Base


def create_model(net_configs: dict, num_of_param: int, dim_of_data: int, 
                 time_points: int, model_type: str, device: torch.device):
    """
    Factory function to create different types of neural network models.
    
    Args:
        net_configs: Configuration dictionary for network architecture. See models/configs for examples.
        num_of_param: Number of parameters to estimate
        dim_of_data: Dimension of input data
        time_points: Number of time points in the sequence
        model_type: Type of model to create ('vae', 'ode_rnn', 'rnn', 'mlp')
        device: PyTorch device to place models on
        
    Returns:
        Configured neural network model
    """
    model_type = model_type.lower()
    latent_dim = net_configs['latent_dim']

    if model_type == 'vae':
        # ===== VAE Model Configuration =====
        decoder_input_dim, decoder_output_dim = latent_dim, num_of_param

        # Configure transformation layer
        hidden_dims = list(net_configs['decoder']['model_hidden_size'])
        hidden_dims = [decoder_input_dim] + hidden_dims + [decoder_output_dim]
        activation_function = net_configs['decoder']['model_nonlinearity']
        transform = build_modules(hidden_dims, activation_function, False)

        # ----- Encoder Configuration -----
        if net_configs['encoder']['type'] == 'ODE':
            # ODE-based encoder
            ode_func_net = create_net(
                latent_dim, latent_dim,
                n_layers=net_configs['encoder']["ode_layers"],
                n_units=net_configs['encoder']['ode_unit'],
                nonlinear=nn.ELU, drop=False
            )
            ode_func = ODEFunc(
                input_dim=latent_dim,
                ode_func_net=ode_func_net,
                device=device
            ).to(device)
            
            diffeq_solver = DiffeqSolver(
                ode_func, "euler", 
                odeint_rtol=1e-3, odeint_atol=1e-5, 
                device=device
            )

            encoder = VAE_ODE_RNN(
                latent_dim, dim_of_data, diffeq_solver, 
                n_GRUUnits=net_configs['encoder']["gru_unit"], 
                device=device
            ).to(device)

        elif net_configs['encoder']['type'] == 'RNN':
            # RNN-based encoder
            encoder = VAE_RNN(
                latent_dim, dim_of_data, GRU_update=None, 
                n_GRUUnits=net_configs['encoder']["gru_unit"], 
                device=device
            ).to(device)

        # ----- Decoder Configuration -----
        if net_configs['decoder']['type'] == 'ODE':

            # ODE-based decoder
            decoder_input_dim = decoder_output_dim
            ode_func_net = create_net(
                decoder_input_dim, decoder_input_dim,
                n_layers=net_configs['decoder']["ode_layers"],
                n_units=net_configs['decoder']['ode_unit'],
                nonlinear=nn.ELU, drop=False
            )
            ode_func = ODEFunc(
                input_dim=decoder_input_dim,
                ode_func_net=ode_func_net,
                device=device
            ).to(device)
            
            diffeq_solver = DiffeqSolver(
                ode_func, "euler", 
                odeint_rtol=1e-3, odeint_atol=1e-5, 
                device=device
            )

            decoder = DecoderODE_RNN(
                decoder_output_dim, dim_of_data, decoder_output_dim, diffeq_solver,
                n_GRUUnits=net_configs['decoder']["gru_unit"], 
                device=device
            ).to(device)

        elif net_configs['decoder']['type'] == 'RNN':
            # RNN-based decoder
            # ERROR: Why does decoder have the shape: decoder_output_dim --> decoder_output_dim?
            decoder = RNN(
                decoder_output_dim, decoder_output_dim, GRU_update=None,
                n_GRUUnits=net_configs['decoder']["gru_unit"], 
                device=device
            )

        return VAE(encoder, decoder, transform)

    elif model_type == 'ode_rnn':
        # ===== ODE-RNN Model Configuration =====
        latent_dim = dim_of_data

        # ----- Encoder Configuration -----
        ode_func_net = create_net(
            latent_dim, latent_dim,
            n_layers=net_configs["ode_layers"],
            n_units=net_configs['ode_unit'],
            nonlinear=nn.ELU, drop=False
        ).to(device)
        
        ode_func = ODEFunc(
            input_dim=latent_dim,
            ode_func_net=ode_func_net,
            device=device
        ).to(device)
        
        diffeq_solver = DiffeqSolver(
            ode_func, "euler", 
            odeint_rtol=1e-6, odeint_atol=1e-8, 
            device=device
        )

        encoder = ODE_RNN(
            latent_dim, dim_of_data, diffeq_solver, 
            GRU_update=None, n_GRUUnits=20, 
            device=device
        )

        # ----- Decoder Configuration -----
        decoder_input_dim = latent_dim + time_points * dim_of_data
        decoder_output_dim = num_of_param
        hidden_dims = [decoder_input_dim, 5, 5, decoder_output_dim]
        
        decoder = DecoderMLP(build_modules(hidden_dims, 'elu', False))

        return EncoderDecoder(encoder, decoder)

    elif model_type == 'rnn':
        # ===== RNN Model Configuration =====
        latent_dim = num_of_param

        # ----- Encoder Configuration -----
        encoder = RNN(
            latent_dim, dim_of_data, GRU_update=None, 
            n_GRUUnits=net_configs["gru_unit"], 
            device=device
        )

        # ----- Decoder Configuration -----
        decoder_input_dim = latent_dim + time_points * dim_of_data
        decoder_output_dim = num_of_param
        hidden_dims = [decoder_input_dim, 5, 5, decoder_output_dim]

        decoder = DecoderMLP(build_modules(hidden_dims, 'elu'))

        return EncoderDecoder(encoder, decoder)
    
    elif model_type == 'mlp':
        # ===== MLP Model Configuration =====
        activation_function = net_configs['model_nonlinearity']

        input_dim = time_points * dim_of_data
        output_dim = num_of_param
        hidden_dims = list(net_configs['model_hidden_size'])
        hidden_dims = [input_dim] + hidden_dims + [output_dim]
        
        mlp = MLP(hidden_dims, activation_function)

        return Base(mlp)

    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'vae', 'ode_rnn', 'rnn', 'mlp'")

    # you may add more models if you wish
