import warnings
warnings.filterwarnings("ignore")


from equations import FitzHugh_Nagumo, Lotka_Volterra, Lorenz
from interface import ODE_Models

from models.DERLPSO import DERLPSO
from models.RLLPSO import RLLPSO
from models.ml_models import MLP, RNN, ODE_RNN, VAE

eval_models = [DERLPSO, MLP, RNN, ODE_RNN, VAE]

points = [5, 10]
interval = [0, 20]
seed = 99

fhn_init_data = [0, 0]
fhn_param_mu = [0.7, 0.8]
fhn_param_sigma = [0.5, 0.5]

fhn = ODE_Models(FitzHugh_Nagumo, fhn_init_data, fhn_param_mu, fhn_param_sigma)
fhn.simulate_data(100, interval, points, seed=100, train=True, over_write=True)
fhn.simulate_data(10, interval, points, seed=101, train=False, over_write=True)
fhn.evaluate(eval_models, interval, points, seed=seed)

# lv_init_data = [0.5, 0.5]
# lv_param_mu = [0.1, 0.02, 0.3,0.1]
# lv_param_sigma = [0.05, 0.01, 0.05, 0.01]

# lv = ODE_Models(Lotka_Volterra, lv_init_data, lv_param_mu, lv_param_sigma)
# lv.simulate_data(100, interval, points, seed=100, train=True, over_write=True)
# lv.simulate_data(10, interval, points, seed=101, train=False, over_write=True)
# lv.evaluate(eval_models, interval, points, seed=seed)

# lz_init_data = [1, 1, 1]
# lz_param_mu = [10, 28, 8/3]
# lz_param_sigma = [1, 1, 1]

# lz = ODE_Models(Lorenz, lz_init_data, lz_param_mu, lz_param_sigma)
# lz.simulate_data(100, interval, points, seed=100, train=True, over_write=True)
# lz.simulate_data(10, interval, points, seed=101, train=False, over_write=True)
# lz.evaluate(eval_models, interval, points, seed=seed)

