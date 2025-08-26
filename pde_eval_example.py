import numpy as np

from Estimators.DERLPSO import DERLPSO
from Estimators.RLLPSO import RLLPSO
from pde_equations import create_heat_function, create_transient_function, create_helmholtz_function
from pde_model import PDE_Models


SEED = 100
NUM_DATA = 2

# HEAT
heat = create_heat_function(nx=5, tx=5, Lx=1, Lt=1)
heat_model = PDE_Models(heat)

heat_α = heat_model.simulate(NUM_DATA, mu=0.5, sigma=0.5, lower=0.0001, upper=1.0, seed=SEED)
data = [heat.f()([a]) for a in heat_α]
test_set = {'param': np.array(heat_α).reshape(-1,1), 'data': np.array(data)} 

r0 = heat_model.evaluate(DERLPSO(heat), test_set, seed = SEED)
heat_model.pprint(r0, "DERLPSO")

r1 = heat_model.evaluate(RLLPSO(heat), test_set, seed = SEED)
heat_model.pprint(r1, "RLLPSO")


## TRANSIENT
transient = create_transient_function(nx=5, steps=5, Lx=1, Lt=1)
transient_model = PDE_Models(transient)

transient_D = transient_model.simulate(NUM_DATA, mu=0.5, sigma=0.5, lower=0.0001, upper=1.0, seed=SEED+1)
transient_v = transient_model.simulate(NUM_DATA, mu=0.5, sigma=0.5, lower=0.0001, upper=1.0, seed=SEED+2)
data = [transient.f()([D, v]) for D, v in zip(transient_D, transient_v)]

test_set = {'param': np.array(list(zip(transient_D, transient_v))), 'data': np.array(data)}

r2 = transient_model.evaluate(DERLPSO(transient), test_set, seed = SEED)
transient_model.pprint(r2, "DERLPSO")

r3 = transient_model.evaluate(RLLPSO(transient), test_set, seed = SEED)
transient_model.pprint(r3, "RLLPSO")

## HELMHOLTZ
helmholtz = create_helmholtz_function(nx=5, ny=5, Lx=1, Ly=1)
helmholtz_model = PDE_Models(helmholtz)

helmholtz_λ = helmholtz_model.simulate(NUM_DATA, mu=0.5, sigma=0.5, lower=0.0001, upper=1.0, seed=SEED+3)
data = [helmholtz.f()([λ]) for λ in helmholtz_λ]
test_set = {'param': np.array(helmholtz_λ).reshape(-1,1), 'data': np.array(data)}

r4 = helmholtz_model.evaluate(DERLPSO(helmholtz), test_set, seed = SEED)
helmholtz_model.pprint(r4, "DERLPSO")

r5 = helmholtz_model.evaluate(RLLPSO(helmholtz), test_set, seed = SEED)
helmholtz_model.pprint(r5, "RLLPSO")
