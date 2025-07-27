import numpy as np
from scipy.integrate import odeint

from .ODE import Model

a = np.random.normal(0.4, 0.8)
b = np.random.normal(1.3, 0.8)
c = np.random.normal(1, 0.8)
d = np.random.normal(1, 0.8)

def lotka_volterra(state, t, α, β, γ, δ):
    l, v = state
    dxdt = α * l - β * l * v
    dydt = δ * l * v - γ * v
    return [dxdt, dydt]

def fitzhugh_nagumo(state, t, a, b, c ,d):
    v, w = state
    dvdt = d * (v - v ** 3 / 3 + w - c)
    dwdt = (-1 / d) * (v - a + b * w)
    return [dvdt, dwdt]

def lorenz(slef, state, t, σ, β, ρ):
        x, y, z = state
        dxdt = σ * (y - x)
        dydt = x * (ρ - z) - y
        dzdt = x * y - β * z
        return [dxdt, dydt, dzdt]

time =  np.arange(0, 4, step=(4 - 0) / 10)
data = odeint(fitzhugh_nagumo, [0, 0], time, args=(a, b, c, d))



model = Model(fitzhugh_nagumo, 4, data, time, threshold=1e-04)
model.initParticles()
model.iterator()



print(f"Param: {np.array([a, b, c, d])}")
print("Best:", ["{:.16f}".format(x) for x in model.getGBest()])
print(f"Fit: {model.getFit()}")