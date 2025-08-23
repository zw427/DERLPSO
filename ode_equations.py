from typing import Callable, List
from inspect import signature
from equation import Equation


class ODE_Equation(Equation):
    def __init__(self, name: str, func: Callable[..., List[float]], num_param: int):
        super().__init__(name, 'ODE', func, num_param)
        self.t_first = list(signature(func).parameters.keys())[0] == 't'


def FitzHugh_Nagumo_func(t, state, param, ξ=-0.4, γ=3.0):
    u, v = state
    θ_0, θ_1 = param
    dudt = γ * (u - u ** 3 / 3 + v + ξ)
    dvdt = - (1 / γ) * (u - θ_0 + θ_1 * v)
    return [dudt, dvdt]

FitzHugh_Nagumo = ODE_Equation("FitzHugh-Nagumo", FitzHugh_Nagumo_func, num_param=2)


def Lotka_Volterra_func(t, state, param):
    x, y = state
    α, β, δ, γ = param
    dxdt = α * x - β * x * y
    dydt = δ * x * y - γ * y
    return [dxdt, dydt]

Lotka_Volterra = ODE_Equation("Lotka-Volterra", Lotka_Volterra_func, num_param=4)


def Lorenz_func(t, state, param):
        x, y, z = state
        σ, β, r = param 
        dxdt = σ * (y - x)
        dydt = r * x - x * z - y
        dzdt = - β * z + x * y 
        return [dxdt, dydt, dzdt]

Lorenz = ODE_Equation("Lorenz", Lorenz_func, num_param=3)
