from typing import Callable, List

class Equation:
    def __init__(self, name: str, de_type: str, func: Callable[..., List[float]]):
        self.name = name
        self.de_type = de_type.upper()
        self.func = func

    def f(self) -> Callable[..., List[float]]:
        return self.func


class ODE_Equation(Equation):
    def __init__(self, name: str, func: Callable[..., List[float]]):
        super().__init__(name, 'ODE', func)
    
    
class PDE_Equation(Equation):
    def __init__(self, name: str, func: Callable[..., List[float]]):
        super().__init__(name, 'PDE', func)


def FitzHugh_Nagumo_func(t, y, param, RI=0.4, tau=3.0):
    v, w = y
    a, b = param
    dvdt = v - v ** 3 / 3 - w + RI
    dwdt = (1 / tau) * (v + a - b * w)
    return [dvdt, dwdt]

FitzHugh_Nagumo = ODE_Equation("FitzHugh-Nagumo", FitzHugh_Nagumo_func)


def Lotka_Volterra_func(t, y, param):
    l, v = y
    a, b, y, d = param
    dldt = a * l - b * l * v
    dvdt = d * l * v - y * v
    return [dldt, dvdt]

Lotka_Volterra = ODE_Equation("Lotka-Volterra", Lotka_Volterra_func)


def Lorenz_func(t, y, param):
        x, y, z = y
        a, b, p = param
        dxdt = a * (y - x)
        dydt = x * (p - z) - y
        dzdt = x * y - b * z
        return [dxdt, dydt, dzdt]

Lorenz = ODE_Equation("Lorenz", Lorenz_func)
