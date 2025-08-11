import warnings
import pprint
# warnings.filterwarnings("ignore")

from torch.distributions import Normal
import torch 

from equation import FitzHugh_Nagumo, Lotka_Volterra, Lorenz
from parameter import Parameter
from interface import ODE_Models

from Estimators.DERLPSO import DERLPSO
# from Estimators.RLLPSO import RLLPSO
# from Estimators.ml_models import MLP, RNN, ODE_RNN, VAE

p0 = Parameter([Normal(torch.Tensor([1, 1]), torch.Tensor([0.4, 0.4]))])

fhn_model = ODE_Models(FitzHugh_Nagumo)
test = fhn_model.simulate(10, parameter=p0, init_data=[0, 0], 
                   interval=[0, 20], point=5, seed=100)

est0 = DERLPSO(FitzHugh_Nagumo)
r0 = fhn_model.evaluate(est0, test['data'], test['time'], test['param'], seed=100)

pprint.pp(r0)