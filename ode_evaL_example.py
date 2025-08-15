import warnings
import pprint
# warnings.filterwarnings("ignore")

from scipy.stats import Normal

from equation import FitzHugh_Nagumo, Lotka_Volterra, Lorenz
from parameter import Parameter
from ode_model import ODE_Models

from Estimators.DERLPSO import DERLPSO
# from Estimators.RLLPSO import RLLPSO
# from Estimators.ml_models import MLP, RNN, ODE_RNN, VAE

# FitzHugh_Nagumo
p0 = Parameter([Normal(mu = [0.7, 0.8], sigma = [0.5, 0.5])])

fhn_model = ODE_Models(FitzHugh_Nagumo)
test0 = fhn_model.simulate(10, parameter=p0, init_data=[0, 0], 
                   interval=[0, 20], point=5, seed=100)

est0 = DERLPSO(FitzHugh_Nagumo)
r0 = fhn_model.evaluate(est0, test0['data'], test0['time'], test0['param'], seed=100)

pprint.pp('##################################')
pprint.pp('FitzHugh_Nagumo')
pprint.pp(r0)
pprint.pp('##################################')
pprint.pp(f'param error mean: {r0["error"].mean(axis=0)}')
pprint.pp(f'MSE mean: {r0["mse"].mean()}')
pprint.pp(f'MSE std: {r0["mse"].std()}')
pprint.pp('##################################\n\n\n\n')

 
p1 = Parameter([Normal(mu=[0.4, 1.3, 1, 1], sigma=[0.5, 0.5, 0.5, 0.5])])

lv_model = ODE_Models(Lotka_Volterra)
test1 = lv_model.simulate(10, parameter=p1, init_data=[0.9, 0.9], 
                   interval=[0, 4], point=5, seed=100)

est1 = DERLPSO(Lotka_Volterra)
r1 = lv_model.evaluate(est1, test1['data'], test1['time'], test1['param'], seed=100)

pprint.pp('##################################')
pprint.pp('Lotka_Volterra')
pprint.pp(r1)
pprint.pp('##################################')
pprint.pp(f'param error mean: {r1["error"].mean(axis=0)}')
pprint.pp(f'MSE mean: {r1["mse"].mean()}')
pprint.pp(f'MSE std: {r1["mse"].std()}')
pprint.pp('##################################\n\n\n\n')

p2 = Parameter([Normal(mu=[2, 4, 1], sigma=[0.5, 0.5, 0.5])])

lz_model = ODE_Models(Lorenz)
test2 = lz_model.simulate(10, parameter=p2, init_data=[0, 1, 1.25], 
                   interval=[0, 4], point=5, seed=100)

est2 = DERLPSO(Lorenz)
r2 = lz_model.evaluate(est2, test2['data'], test2['time'], test2['param'], seed=100)

pprint.pp('##################################')
pprint.pp('Lorenz')
pprint.pp(r2)
pprint.pp('##################################')
pprint.pp(f'param error mean: {r2["error"].mean(axis=0)}')
pprint.pp(f'MSE mean: {r2["mse"].mean()}')
pprint.pp(f'MSE std: {r2["mse"].std()}')
pprint.pp('##################################\n\n\n\n')