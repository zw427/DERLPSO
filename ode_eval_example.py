import time

from scipy.stats import Normal, Uniform
import numpy as np

from ode_equations import FitzHugh_Nagumo, Lotka_Volterra, Lorenz
from ode_model import ODE_Models
from parameter import Parameter

from Estimators.DERLPSO import DERLPSO
from Estimators.RLLPSO import RLLPSO
from Estimators.ml_estimator import MLP, RNN, ODE_RNN, VAE

MULTIPLIER = 100
OUTPUT_TEXT = "ode_eval_results.txt"

def run_eval(func, config, num_data, num_state, parameter, init_data, interval, points, seed):
    for point in points:
        model = ODE_Models(func)

        train = model.simulate(num_data * MULTIPLIER, parameter=parameter, init_data=init_data, 
                               interval=interval, point=point, seed=seed + 1)
        test = model.simulate(num_data, parameter=parameter, init_data=init_data, 
                                   interval=interval, point=point, seed=seed)

        np.random.seed(int(time.time()) % 2**32)

        if True:
            r0 = model.evaluate(DERLPSO(func), test, train)
            model.pprint(r0, "DERLPSO", OUTPUT_TEXT)

        # RLLPSO
        if True:
            r1 = model.evaluate(RLLPSO(func), test, train)
            model.pprint(r1, "RLLPSO", OUTPUT_TEXT)

        # MLP
        if True:
            r2 = model.evaluate(MLP(func.num_param, num_state, point, config), test, train)
            model.pprint(r2, "MLP", OUTPUT_TEXT)

        # RNN
        if True:
            r3 = model.evaluate(RNN(func.num_param, num_state, point, config), test, train)
            model.pprint(r3, "RNN", OUTPUT_TEXT)

        # ODE_RNN 
        if True:
            r4 = model.evaluate(ODE_RNN(func.num_param, num_state, point, config), test, train)
            model.pprint(r4, "ODE_RNN", OUTPUT_TEXT)
            
        # VAE
        if True:
            r5 = model.evaluate(VAE(func.num_param, num_state, point, config), test, train)
            model.pprint(r5, "VAE", OUTPUT_TEXT)


if __name__ == "__main__":

    θ_0 = Normal(mu = 0.7, sigma = 0.5)
    θ_1 = Normal(mu = 0.8, sigma = 0.5)
    p0 = Parameter([θ_0, θ_1])

    run_eval(func = FitzHugh_Nagumo, config = "Estimators/configs/fn.yaml", num_data = 10, num_state = 2, 
             parameter = p0, init_data = [0, 0], interval = [0, 20], points = [5], seed = 100)

    α = Normal(mu = 0.4, sigma = 0.5)
    β = Normal(mu = 1.3, sigma = 0.5)
    δ = Normal(mu = 1, sigma = 0.5)
    γ = Normal(mu = 1, sigma = 0.5)
    p1 = Parameter([α, β, δ, γ])
    run_eval(func = Lotka_Volterra, config = "Estimators/configs/lovo.yaml", num_data = 10, num_state = 2, 
             parameter = p1, init_data = [0.9, 0.9], interval = [0, 4], points = [5], seed = 100)

    σ = Normal(mu=2, sigma=0.5)
    β = Normal(mu=4, sigma=0.5)
    r = Normal(mu=1, sigma=0.5)
    p2 = Parameter([σ, β, r])
    run_eval(func = Lorenz, config = "Estimators/configs/lv.yaml", num_data = 10, num_state = 3, 
             parameter = p2, init_data = [0, 1, 1.25], interval = [0, 4], points = [5], seed = 100)
