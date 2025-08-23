from typing import Callable, List
from equation import Equation

import numpy as np
from fipy import Grid1D, Grid2D, CellVariable, DiffusionTerm, TransientTerm, ImplicitSourceTerm, LinearLUSolver,HybridConvectionTerm


class PDE_Equation(Equation):
    def __init__(self, name: str, func: Callable[..., List[float]], num_param: int):
        super().__init__(name, 'PDE', func, num_param)
        self.num_param = num_param


def create_heat_function(nx=5, tx=5, Lx=1.0, Lt=1.0):
    """
    Create a heat equation function with specified grid and time parameters.
    
    Args:
        nx: Number of spatial grid points
        tx: Number of time steps
        Lx: Spatial domain length
        Lt: Total time duration
    
    Returns:
        Function that solves heat equation with given parameters
    """
    def heat(X):
        alpha = X[0]
        mesh = Grid1D(nx=nx, Lx=Lx)
        T = CellVariable(name="temperature", mesh=mesh, value=0.0, hasOld=True)
        
        # Create initial condition based on nx
        if nx == 20:
            # Use the original initial condition for nx=20
            initial_values = [0.1, 0.2, 0.5, 1, 1, 1, 1, 1, 0.5, 0.2, 0.2, 0.5, 1, 1, 1, 1, 1, 0.5, 0.2, 0.1]
        else:
            # Create a scaled initial condition for different nx
            center = nx // 2
            initial_values = []
            for i in range(nx):
                # Create a bell-shaped initial condition
                distance = abs(i - center) / (nx / 20.0)  # Scale to original spacing
                if distance < 2:
                    initial_values.append(1.0)
                elif distance < 4:
                    initial_values.append(0.5)
                elif distance < 6:
                    initial_values.append(0.2)
                else:
                    initial_values.append(0.1)
        
        T.setValue([initial_values])
        T.constrain(0.0, mesh.facesLeft)
        T.constrain(0.0, mesh.facesRight)
        
        eq = TransientTerm() == DiffusionTerm(coeff=alpha)
        dt = Lt / tx
        u = np.zeros((tx + 1, nx))
        u[0,] = T.value
        
        for step in range(tx):
            T.updateOld()
            eq.solve(var=T, dt=dt)
            u[step + 1,] = T.value
        
        return u
    
    return PDE_Equation("Heat", heat, 1)


def create_transient_function(nx=5, steps=None, Lx=1.0, Lt=1.0):
    """
    Create a transient convection-diffusion equation function with specified parameters.
    
    Args:
        nx: Number of spatial grid points
        steps: Number of time steps (default: nx)
        Lx: Spatial domain length
        total_time: Total time duration
    
    Returns:
        Function that solves transient convection-diffusion equation with given parameters
    """
    if steps is None:
        steps = nx
        
    def transient_custom(X):
        diffCoeff = X[0][0]
        convCoeff = X[0][1]
        mesh = Grid1D(dx=Lx / nx, nx=nx)
        timeStepDuration = Lt / steps
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
            eqn.solve(var=phi, dt=timeStepDuration)
            u[step + 1,] = phi.value

        return u
    
    return PDE_Equation("Transient Convection-Diffusion", transient_custom, 2)


def create_helmholtz_function(nx=5, ny=5, Lx=1.0, Ly=1.0):
    """
    Create a Helmholtz equation function with specified grid parameters.
    
    Args:
        nx: Number of grid points in x direction
        ny: Number of grid points in y direction
        Lx: Domain length in x direction
        Ly: Domain length in y direction
    
    Returns:
        Function that solves Helmholtz equation with given parameters
    """
    def helmholtz_custom(X):
        wavelength = X[0]
        dx = Lx / nx
        dy = Ly / ny
        mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
        u = CellVariable(name="u", mesh=mesh, value=1.0)
        
        # Create initial condition based on grid size
        # Create a cross pattern scaled to the grid size
        initial_values = np.zeros(nx * ny)
        center_x = nx // 2
        center_y = ny // 2
        
        for i in range(ny):
            for j in range(nx):
                idx = i * nx + j
                # Create vertical line at center
                if j == center_x and (i < center_y - 1 or i > center_y + 1):
                    initial_values[idx] = 1.0
                # Create horizontal gap at center
                elif i == center_y and j != center_x:
                    initial_values[idx] = 0.0
                else:
                    initial_values[idx] = 0.0
        
        u.setValue(initial_values)
        k = 2 * np.pi / wavelength
        helmholtz_eq = DiffusionTerm(coeff=1.0) + ImplicitSourceTerm(coeff=k ** 2)
        u.constrain(0.0, mesh.facesLeft)
        u.constrain(0.0, mesh.facesRight)
        u.constrain(0.0, mesh.facesBottom)
        u.constrain(0.0, mesh.facesTop)
        solver = LinearLUSolver()
        helmholtz_eq.solve(var=u, solver=solver)
        return u
    
    return PDE_Equation("Helmholtz", helmholtz_custom, 1)
