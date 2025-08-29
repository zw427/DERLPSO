import numpy as np
import time

from sde_equations import (
    Ornstein_Uhlenbeck, Geometric_Brownian_Motion, Cox_Ingersoll_Ross,
    Heston_Model, Lorenz_Model
)
from sde_model import SDE_Models
from Estimators.SDE_Estimator import SDE_Estimator

OUTPUT_TEXT = "sde_eval_results.txt"

def run_single_test(equation, true_params, t_span, dt, estimator_config):
    model = SDE_Models(equation)
    time_points = np.arange(t_span[0], t_span[1] + dt, dt)
    
    n_trajectories = 5
    trajectories = [
        model._solve_sde(true_params, equation.initial_conditions, time_points, seed=42 + i) 
        for i in range(n_trajectories)
    ]
    
    initial_conditions_np = np.atleast_1d(equation.initial_conditions)

    if initial_conditions_np.ndim == 1 and initial_conditions_np.shape[0] == 1:
        synthetic_data_concatenated = np.concatenate(trajectories).flatten()
    else:
        dimension = initial_conditions_np.shape[0]
        concatenated_components = []
        for dim in range(dimension):
            component_data = np.concatenate([traj[:, dim] for traj in trajectories])
            concatenated_components.append(component_data)
        synthetic_data_concatenated = np.concatenate(concatenated_components)

    test_data = synthetic_data_concatenated[np.newaxis, :]
    test_time = time_points[np.newaxis, :] 

    estimator = SDE_Estimator(
        func=equation,
        particle_num=estimator_config.get('particle_num', 100),
        max_iter=estimator_config.get('max_iter', 200),
        grid_resolution=estimator_config.get('grid_resolution', 5)
    )
    estimated_params_array = estimator.predict(test_data, test_time)

    param_names = list(equation.param_bounds.keys())
    true_params_array = np.array([true_params[name] for name in param_names])[np.newaxis, :]
    
    est_params_dict = dict(zip(param_names, estimated_params_array[0]))
    estimated_data_concatenated = estimator._sde_prediction_concatenated(est_params_dict, time_points, seed_base=42)

    results = {
        'data': test_data, 'param': true_params_array, 'time': test_time,
        'prediction': estimated_params_array, 'error': true_params_array - estimated_params_array,
        'mse': np.array([np.mean((synthetic_data_concatenated - estimated_data_concatenated)**2)])
    }
    model.pprint(results, "SDE_Estimator", file=OUTPUT_TEXT)


if __name__ == "__main__":
    full_config = {'particle_num': 100, 'max_iter': 200, 'grid_resolution': 5}
    
    run_single_test(
        equation=Ornstein_Uhlenbeck,
        true_params={'theta': 1.0, 'mu': 0.1, 'sigma': 0.1},
        t_span=(0, 5), dt=0.005,
        estimator_config=full_config
    )
    run_single_test(
        equation=Geometric_Brownian_Motion,
        true_params={'mu': 0.05, 'sigma': 0.2},
        t_span=(0, 1), dt=0.01,
        estimator_config=full_config
    )
    run_single_test(
        equation=Cox_Ingersoll_Ross,
        true_params={'kappa': 1.5, 'theta': 0.04, 'sigma': 0.2},
        t_span=(0, 1), dt=0.01,
        estimator_config=full_config
    )
    run_single_test(
        equation=Heston_Model,
        true_params={'mu': 0.05, 'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3},
        t_span=(0, 1), dt=0.01,
        estimator_config=full_config
    )
    run_single_test(
        equation=Lorenz_Model,
        true_params={'s': 10.0, 'r': 28.0, 'b': 8.0 / 3.0, 'sigma': 2.0},
        t_span=(0, 1), dt=0.01,
        estimator_config=full_config
    )