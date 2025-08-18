from scipy.stats import Normal, Uniform

from ode_equations import FitzHugh_Nagumo, Lotka_Volterra, Lorenz
from ode_model import ODE_Models
from parameter import Parameter

from Estimators.DERLPSO import DERLPSO
# from Estimators.RLLPSO import RLLPSO
from Estimators.ml_estimator import MLP, RNN, ODE_RNN, VAE

def evaluate_FitzHugh_Nagumo(num_data, points, seed):
    # parameters
    θ_0 = Normal(mu = 0.7, sigma = 0.5)
    θ_1 = Normal(mu = 0.8, sigma = 0.5)
    p0 = Parameter([θ_0, θ_1])

    for point in points: 
        fhn_model = ODE_Models(FitzHugh_Nagumo)
        train = fhn_model.simulate(num_data, parameter=p0, init_data=[0, 0], 
                                   interval=[0, 20], point=point, seed=seed + 1)
        test = fhn_model.simulate(num_data, parameter=p0, init_data=[0, 0], 
                                   interval=[0, 20], point=point, seed=seed)

        # DERLPSO
        if False:
            est0 = DERLPSO(FitzHugh_Nagumo)
            r0 = fhn_model.evaluate(est0, test, train, seed)
            fhn_model.pprint(r0)

        # MLP
        if True:
            est1 = MLP(FitzHugh_Nagumo.num_param, 2, point, "Estimators/configs/fn.yaml")
            r1 = fhn_model.evaluate(est1, test, train, seed)
            fhn_model.pprint(r1)

        # RNN
        if True:
            est2 = RNN(FitzHugh_Nagumo.num_param, 2, point, "Estimators/configs/fn.yaml")
            r2 = fhn_model.evaluate(est2, test, train, seed)
            fhn_model.pprint(r2)


        # ODE_RNN 
        if True:
            est3 = ODE_RNN(FitzHugh_Nagumo.num_param, 2, point, "Estimators/configs/fn.yaml")
            r3 = fhn_model.evaluate(est3, test, train, seed)
            fhn_model.pprint(r3)

        # VAE
        if True:
            est4 = VAE(FitzHugh_Nagumo.num_param, 2, point, "Estimators/configs/fn.yaml")
            r4 = fhn_model.evaluate(est4, test, train, seed)
            fhn_model.pprint(r4)


# p1 = Parameter([Normal(mu=[0.4, 1.3, 1, 1], sigma=[0.5, 0.5, 0.5, 0.5])])

# lv_model = ODE_Models(Lotka_Volterra)
# test1 = lv_model.simulate(10, parameter=p1, init_data=[0.9, 0.9], 
#                    interval=[0, 4], point=5, seed=100)

# est1 = DERLPSO(Lotka_Volterra)
# r1 = lv_model.evaluate(est1, test1['data'], test1['time'], test1['param'], seed=100)



# p2 = Parameter([Normal(mu=[2, 4, 1], sigma=[0.5, 0.5, 0.5])])

# lz_model = ODE_Models(Lorenz)
# test2 = lz_model.simulate(10, parameter=p2, init_data=[0, 1, 1.25], 
#                    interval=[0, 4], point=5, seed=100)

# est2 = DERLPSO(Lorenz)
# r2 = lz_model.evaluate(est2, test2['data'], test2['time'], test2['param'], seed=100)


if __name__ == "__main__":
    # evaluate_FitzHugh_Nagumo(100, [5, 10, 20])
    evaluate_FitzHugh_Nagumo(10, [5], 100)
