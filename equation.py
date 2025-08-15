from typing import Callable, List
from inspect import signature

class Equation:
    def __init__(self, name: str, de_type: str, func: Callable[..., List[float]], num_param: int):
        self.name = name
        self.de_type = de_type.upper()
        self.func = func
        self.num_param = num_param

    def f(self) -> Callable[..., List[float]]:
        return self.func


class ODE_Equation(Equation):
    def __init__(self, name: str, func: Callable[..., List[float]], num_param: int):
        super().__init__(name, 'ODE', func, num_param)
        self.t_first =  list(signature(func).parameters.keys())[0] == 't'


class PDE_Equation(Equation):
    def __init__(self, name: str, func: Callable[..., List[float]], num_param: int):
        super().__init__(name, 'PDE', func, num_param)


def FitzHugh_Nagumo_func(t, state, param, ξ=-0.4, γ=3.0):
    u, v = state
    θ_0, θ_1 = param
    dudt = γ * (u - u ** 3 / 3 + v + ξ)
    dvdt = - (1 / γ) * (u - θ_0 + θ_1 * v)
    return [dudt, dvdt]

FitzHugh_Nagumo = ODE_Equation("FitzHugh-Nagumo", FitzHugh_Nagumo_func, 2)


def Lotka_Volterra_func(t, state, param):
    x, y = state
    α, β, δ, γ = param
    dxdt = α * x - β * x * y
    dydt = δ * x * y - γ * y
    return [dxdt, dydt]

Lotka_Volterra = ODE_Equation("Lotka-Volterra", Lotka_Volterra_func, 4)


def Lorenz_func(t, state, param):
        x, y, z = state
        σ, β, r = param 
        dxdt = σ * (y - x)
        dydt = r * x - x * z - y
        dzdt = - β * z + x * y 
        return [dxdt, dydt, dzdt]

Lorenz = ODE_Equation("Lorenz", Lorenz_func, 3)
