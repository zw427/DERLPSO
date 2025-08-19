from scipy.stats import Normal, Uniform

from ode_equations import FitzHugh_Nagumo, Lotka_Volterra, Lorenz
from ode_model import ODE_Models
from parameter import Parameter

from Estimators.DERLPSO import DERLPSO
from Estimators.RLLPSO import RLLPSO
from Estimators.ml_estimator import MLP, RNN, ODE_RNN, VAE



def run_eval(func, config, num_data, num_state, parameter, init_data, interval, points, seed):
    for point in points:
        model = ODE_Models(func)

        train = model.simulate(num_data * 100, parameter=parameter, init_data=init_data, 
                               interval=interval, point=point, seed=seed + 1)
        test = model.simulate(num_data, parameter=parameter, init_data=init_data, 
                                   interval=interval, point=point, seed=seed)

        print(f"function: {func.name}")

        if True:
            r0 = model.evaluate(DERLPSO(func), test, train, seed)
            print("DERLPSO")
            model.pprint(r0)

        # RLLPSO
        if True:
            r1 = model.evaluate(RLLPSO(func), test, train, seed)
            print("RLLPSO")
            model.pprint(r1)

        # MLP
        if True:
            r2 = model.evaluate(MLP(func.num_param, num_state, point, config), test, train, seed)
            print("MLP")
            model.pprint(r2)

        # RNN
        if True:
            r3 = model.evaluate(RNN(func.num_param, num_state, point, config), test, train, seed)
            print("RNN")
            model.pprint(r3)


        # ODE_RNN 
        if True:
            r4 = model.evaluate(ODE_RNN(func.num_param, num_state, point, config), test, train, seed)
            print("ODE_RNN")
            model.pprint(r4)
            
        # VAE
        if True:
            r5 = model.evaluate(VAE(func.num_param, num_state, point, config), test, train, seed)
            print("VAE")
            model.pprint(r5)


if __name__ == "__main__":

    θ_0 = Normal(mu = 0.7, sigma = 0.5)
    θ_1 = Normal(mu = 0.8, sigma = 0.5)
    p0 = Parameter([θ_0, θ_1])

    run_eval(FitzHugh_Nagumo, "Estimators/configs/fn.yaml", 10, 2, p0, [0, 0], [0, 20], [5], 100)

    α = Normal(mu = 0.4, sigma = 0.5)
    β = Normal(mu = 1.3, sigma = 0.5)
    δ = Normal(mu = 1, sigma = 0.5)
    γ = Normal(mu = 1, sigma = 0.5)
    p1 = Parameter([α, β, δ, γ])
    run_eval(Lotka_Volterra, "Estimators/configs/lovo.yaml", 10, 2, p1, [0.9, 0.9], [0, 4], [5], 100)

    σ = Normal(mu=2, sigma=0.5)
    β = Normal(mu=4, sigma=0.5)
    r = Normal(mu=1, sigma=0.5)

    p2 = Parameter([σ, β, r])
    run_eval(Lorenz, "Estimators/configs/lv.yaml", 10, 3, p2, [0, 1, 1.25], [0, 4], [5], 100)
    