from eval import *

name = "fitzhugh_nagumo"
func = 'Estimation.Ode_Equation.FN_equation'
init_data = [0, 0]
param_mu = [0.7, 0.8]
param_sigma = [0.5, 0.5]

# models = ['DERLPSO', 'RLLPSO', 'MLP', 'RNN', 'ODE_RNN', 'VAE']
models = ['DERLPSO', 'MLP', 'RNN', 'ODE_RNN', 'VAE']
# points = [5, 10, 20, 40]
points = [5, 10]
interval = [0, 20]
seed = 100

ode = ODE(name, func, init_data, param_mu, param_sigma)
evalu = ODE_Evaluate(ode, models, points, interval, seed)
evalu.simulate_train_data(num_data = 10)
evalu.train("Estimation/configs/fn.yaml")

evalu.simulate_test_data(num_data = 2)
evalu.test()