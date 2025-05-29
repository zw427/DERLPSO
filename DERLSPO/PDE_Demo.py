
import numpy as np
from fipy import Grid1D, CellVariable, DiffusionTerm, TransientTerm
from scipy.stats import truncnorm

from PDE import Model

def heat(X):
    alpha = X[0]
    nx = 20
    Lx = 1.0
    mesh = Grid1D(nx=nx, Lx=Lx)
    T = CellVariable(name="temperature", mesh=mesh, value=0.0, hasOld=True)
    T.setValue([[0.1, 0.2, 0.5, 1, 1, 1, 1, 1, 0.5, 0.2, 0.2, 0.5, 1, 1, 1, 1, 1, 0.5, 0.2, 0.1]])
    T.constrain(0.0, mesh.facesLeft)
    T.constrain(0.0, mesh.facesRight)
    eq = TransientTerm() == DiffusionTerm(coeff=alpha)
    Lt = 1.0
    tx = 20
    dt = Lt / tx
    u = np.zeros((tx + 1, nx))
    u[0,] = T.value
    for step in range(tx):
        T.updateOld()
        eq.solve(var=T, dt=dt)
        u[step + 1,] = T.value
    return u


def helmholtz(X):
    len = X[0]
    nx = 5
    ny = 5
    Lx = 1.0
    Ly = 1.0
    dx = Lx / nx
    dy = Ly / ny
    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    u = CellVariable(name="u", mesh=mesh, value=1.0)
    u.setValue(
        [0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 1, 0, 0,
         ])
    k = 2 * np.pi / len
    helmholtz_eq = DiffusionTerm(coeff=1.0) + ImplicitSourceTerm(coeff=k ** 2)
    u.constrain(0.0, mesh.facesLeft)
    u.constrain(0.0, mesh.facesRight)
    u.constrain(0.0, mesh.facesBottom)
    u.constrain(0.0, mesh.facesTop)
    solver = LinearLUSolver()
    helmholtz_eq.solve(var=u, solver=solver)
    return u

def transient(X):
    diffCoeff = X[0
    convCoeff = (X[1],)
    Lx = 1.
    nx = 5
    mesh = Grid1D(dx=Lx / nx, nx=nx)
    total_time = 1.
    steps = nx
    timeStepDuration = total_time / steps
    f_phi0 = lambda x: (20 * x - 11) / 9

    pos = mesh.x.value
    phi = CellVariable(mesh=mesh, value=f_phi0(pos), hasOld=1, name='phi')

    phi.constrain(0., mesh.facesLeft)
    phi.constrain(0., mesh.facesRight)

    u = np.zeros((steps+1, nx))
    u[0,] = phi.value

    eqn = TransientTerm() + HybridConvectionTerm(coeff=convCoeff) == DiffusionTerm(coeff=diffCoeff)

    for step in range(steps):
        phi.updateOld()
        eqn.solve(var=phi,
                 dt=timeStepDuration)
        u[step + 1,] = phi.value

    return u

mu = 0.5
sigma = 0.5
lower, upper = 0.0001, 1
a, b = (lower - mu) / sigma, (upper - mu) / sigma
truncated_normal_dist = truncnorm(a, b, loc=mu, scale=sigma)
a = truncated_normal_dist.rvs()

data = heat([a])

model = Model(heat, 1, data)
model.initParticles()
model.iterator()

print(f"Param: {[a]}")
print("Best:", ["{:.16f}".format(x) for x in model.getGBest()])
print(f"Fit: {model.getFit()}")




