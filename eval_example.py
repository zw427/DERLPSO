
from equations import FitzHugh_Nagumo, Lotka_Volterra, Lorenz
from interface import ODE_Models
from models.base_model import BaseModel
from models.ml_models import MLModel
init_data = [0, 0]
param_mu = [0.7, 0.8]
param_sigma = [0.5, 0.5]

eval_models = [BaseModel, MLModel]
points = [5, 10]
interval = [0, 20]
seed = 99

fhn = ODE_Models(FitzHugh_Nagumo, init_data, param_mu, param_sigma)
fhn.simulate_data(10000, interval, points, seed=100, train=True, over_write=True)
fhn.simulate_data(1000, interval, points, seed=101, train=False, over_write=True)
# fhn.evaluate(eval_models, points, interval, seed=seed)

lv = ODE_Models(Lotka_Volterra, init_data, param_mu, param_sigma)
lv.simulate_data(10000, interval, points, seed=100, train=True, over_write=True)
# lv.simulate_data(1000, interval, points, seed=101, train=False, over_write=True)
# # lv.evaluate(eval_models, points, interval, seed=seed)

lz = ODE_Models(Lorenz, init_data, param_mu, param_sigma)
lz.simulate_data(10000, interval, points, seed=100, train=True, over_write=True)
# lz.simulate_data(1000, interval, points, seed=101, train=False, over_write=True)
# lz.evaluate(eval_models, points, interval, seed=seed)
