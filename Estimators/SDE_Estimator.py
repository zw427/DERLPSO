import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import itertools

from Estimators.estimator import Estimator
from sde_equations import SDE_Equation
from sde_model import SDE_Models


class SDE_Estimator(Estimator):
    def __init__(
        self, func: SDE_Equation, particle_num=100, max_iter=200, grid_resolution=5
    ):
        if not isinstance(func, SDE_Equation):
            raise TypeError("SDE_Estimator only works with SDE_Equation objects.")

        self.func = func
        self.sde_model = SDE_Models(func)
        self.particleNum = particle_num
        self.maxIter = max_iter
        self.threshold = 1e-6
        self.grid_resolution = grid_resolution
        self.n_sim_for_fitness = 5

    def _sde_prediction_concatenated(
        self, params_dict: dict, time_points: np.ndarray, seed_base: int
    ) -> np.ndarray:
        sim_data_list = [
            self.sde_model._solve_sde(
                params_dict,
                self.func.initial_conditions,
                time_points,
                seed=seed_base + i,
            )
            for i in range(self.n_sim_for_fitness)
        ]
        initial_conditions_np = np.atleast_1d(self.func.initial_conditions)
        if initial_conditions_np.size == 1:
            return np.concatenate([t.flatten() for t in sim_data_list])
        else:
            dimension = initial_conditions_np.shape[0]
            return np.concatenate(
                [
                    np.concatenate([traj[:, dim] for traj in sim_data_list])
                    for dim in range(dimension)
                ]
            )

    def _fitness_function(
        self, parameters: np.ndarray, target_data: np.ndarray, time_points: np.ndarray
    ) -> float:
        param_names = list(self.func.param_bounds.keys())
        params_dict = dict(zip(param_names, parameters))
        try:
            predicted_data = self._sde_prediction_concatenated(
                params_dict, time_points, seed_base=42
            )
            if predicted_data.shape != target_data.shape:
                return float("inf")
            residuals = target_data - predicted_data
            if len(residuals) < 4:
                return np.mean(residuals**2)
            _, p_value = stats.shapiro(residuals)
            normality_penalty = -np.log(p_value + 1e-10)
            mse = np.mean(residuals**2)
            return mse * (1 + 0.1 * normality_penalty)
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")

    def _pso_optimizer(
        self, target_data: np.ndarray, time_points: np.ndarray, fixed_params: dict
    ) -> tuple:
        opt_param_names = [
            name for name in self.func.param_bounds if name not in fixed_params
        ]
        if not opt_param_names:
            return fixed_params, float("inf")
        n_params = len(opt_param_names)
        bounds = np.array([self.func.param_bounds[name] for name in opt_param_names])
        lower, upper = bounds[:, 0], bounds[:, 1]
        X = lower + np.random.rand(self.particleNum, n_params) * (upper - lower)
        V = np.zeros_like(X)
        pBest, pFit = X.copy(), np.full(self.particleNum, float("inf"))
        gBest, gFit = np.zeros(n_params), float("inf")

        for i in range(self.particleNum):
            params_dict = {**fixed_params, **dict(zip(opt_param_names, X[i]))}
            param_array = [params_dict[name] for name in self.func.param_bounds.keys()]
            pFit[i] = self._fitness_function(
                np.array(param_array), target_data, time_points
            )
            if pFit[i] < gFit:
                gFit, gBest = pFit[i], X[i].copy()

        for iteration in range(self.maxIter):
            w = 0.9 - 0.5 * (iteration / self.maxIter)
            c1 = 2.0 - 0.5 * (iteration / self.maxIter)
            c2 = 0.5 + 1.5 * (iteration / self.maxIter)

            for i in range(self.particleNum):
                r1, r2 = np.random.rand(2, n_params)
                V[i] = w * V[i] + c1 * r1 * (pBest[i] - X[i]) + c2 * r2 * (gBest - X[i])
                X[i] = np.clip(X[i] + V[i], lower, upper)

                params_dict = {**fixed_params, **dict(zip(opt_param_names, X[i]))}
                param_array = [
                    params_dict[name] for name in self.func.param_bounds.keys()
                ]
                fitness = self._fitness_function(
                    np.array(param_array), target_data, time_points
                )

                if fitness < pFit[i]:
                    pFit[i], pBest[i] = fitness, X[i].copy()
                    if fitness < gFit:
                        gFit, gBest = fitness, X[i].copy()

            if gFit < self.threshold:
                break

        return {**fixed_params, **dict(zip(opt_param_names, gBest))}, gFit

    def train(self, train_data, seed=None):
        pass

    def predict(
        self, data: np.ndarray, time: np.ndarray, seed: int = None
    ) -> np.ndarray:
        num_samples = data.shape[0]
        all_predictions = np.zeros((num_samples, self.func.num_param))
        param_names_ordered = list(self.func.param_bounds.keys())
        for i in range(num_samples):
            sample_data, sample_time = data[i], time[i]
            noise_param_names = self.func.noise_param_names
            if not noise_param_names:
                best_params_dict, _ = self._pso_optimizer(sample_data, sample_time, {})
            else:
                noise_grids = {
                    name: np.linspace(
                        *self.func.param_bounds[name], self.grid_resolution
                    )
                    for name in noise_param_names
                }
                grid_combinations = [
                    dict(zip(noise_grids.keys(), combo))
                    for combo in itertools.product(*noise_grids.values())
                ]
                best_fitness, best_params_dict = float("inf"), None
                for fixed_params in tqdm(
                    grid_combinations, desc=f"Grid Search (Sample {i + 1})"
                ):
                    params_dict, fitness = self._pso_optimizer(
                        sample_data, sample_time, fixed_params
                    )
                    if fitness < best_fitness:
                        best_fitness, best_params_dict = fitness, params_dict
            all_predictions[i, :] = [
                best_params_dict[name] for name in param_names_ordered
            ]
        return all_predictions
