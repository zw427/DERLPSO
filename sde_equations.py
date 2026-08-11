from typing import Callable, Dict, Union, List
import numpy as np
from equation import Equation


class SDE_Equation(Equation):
    def __init__(
        self,
        name: str,
        drift_template: Callable,
        diffusion_template: Callable,
        num_param: int,
        param_bounds: Dict[str, tuple],
        initial_conditions: Union[float, np.ndarray],
        noise_param_names: List[str] = None,
        correlation_matrix: np.ndarray = None,
    ):
        super().__init__(name, "SDE", drift_template, num_param)
        self.drift_template = drift_template
        self.diffusion_template = diffusion_template
        self.param_bounds = param_bounds
        self.initial_conditions = initial_conditions
        self.noise_param_names = noise_param_names or []
        self.correlation_matrix = correlation_matrix


# 1. Ornstein-Uhlenbeck (OU)
def ou_drift_template(x, t, params):
    return params["theta"] * (params["mu"] - x)


def ou_diffusion_template(x, t, params):
    return params["sigma"]


Ornstein_Uhlenbeck = SDE_Equation(
    name="Ornstein-Uhlenbeck",
    drift_template=ou_drift_template,
    diffusion_template=ou_diffusion_template,
    num_param=3,
    param_bounds={"theta": (0.1, 2.0), "mu": (0, 1.0), "sigma": (0.1, 1.0)},
    initial_conditions=1.0,
    noise_param_names=["sigma"],
)


# 2. Geometric Brownian Motion (GBM)
def gbm_drift_template(x, t, params):
    return params["mu"] * x


def gbm_diffusion_template(x, t, params):
    return params["sigma"] * x


Geometric_Brownian_Motion = SDE_Equation(
    name="Geometric-Brownian-Motion",
    drift_template=gbm_drift_template,
    diffusion_template=gbm_diffusion_template,
    num_param=2,
    param_bounds={"mu": (0.01, 0.2), "sigma": (0.1, 0.5)},
    initial_conditions=100.0,
    noise_param_names=["sigma"],
)


# 3. Cox-Ingersoll-Ross (CIR)
def cir_drift_template(r, t, params):
    return params["kappa"] * (params["theta"] - r)


def cir_diffusion_template(r, t, params):
    return params["sigma"] * np.sqrt(max(r, 1e-8))


Cox_Ingersoll_Ross = SDE_Equation(
    name="Cox-Ingersoll-Ross",
    drift_template=cir_drift_template,
    diffusion_template=cir_diffusion_template,
    num_param=3,
    param_bounds={"kappa": (0.1, 3.0), "theta": (0.01, 0.1), "sigma": (0.05, 0.5)},
    initial_conditions=0.05,
    noise_param_names=["sigma"],
)


# 4. Heston
def heston_drift_template(X, t, params):
    S, V = X[0], X[1]
    dS_dt = params["mu"] * S
    dV_dt = params["kappa"] * (params["theta"] - V)
    return np.array([dS_dt, dV_dt])


def heston_diffusion_template(X, t, params):
    S, V = X[0], X[1]
    V = max(V, 1e-8)
    sqrt_V = np.sqrt(V)
    return np.array([[sqrt_V * S, 0], [0, params["sigma"] * sqrt_V]])


heston_rho = -0.7
heston_corr_matrix = np.array([[1.0, heston_rho], [heston_rho, 1.0]])

Heston_Model = SDE_Equation(
    name="Heston-Model",
    drift_template=heston_drift_template,
    diffusion_template=heston_diffusion_template,
    num_param=4,
    param_bounds={
        "mu": (0.01, 0.15),
        "kappa": (0.5, 5.0),
        "theta": (0.01, 0.1),
        "sigma": (0.1, 0.8),
    },
    initial_conditions=np.array([100.0, 0.04]),
    noise_param_names=["sigma"],
    correlation_matrix=heston_corr_matrix,
)


# 5. Lorenz 63 System (Stochastic)
def lorenz_drift_template(X, t, params):
    x, y, z = X[0], X[1], X[2]
    dx_dt = -params["s"] * x + params["s"] * y
    dy_dt = -x * z + params["r"] * x - y
    dz_dt = x * y - params["b"] * z
    return np.array([dx_dt, dy_dt, dz_dt])


def lorenz_diffusion_template(X, t, params):
    return np.eye(3) * params["sigma"]


Lorenz_Model = SDE_Equation(
    name="Lorenz-63-Stochastic",
    drift_template=lorenz_drift_template,
    diffusion_template=lorenz_diffusion_template,
    num_param=4,
    param_bounds={
        "s": (5.0, 15.0),
        "r": (20.0, 40.0),
        "b": (1.5, 5.0),
        "sigma": (1.0, 5.0),
    },
    initial_conditions=np.array([1.0, 1.0, 20.0]),
    noise_param_names=["sigma"],
)
