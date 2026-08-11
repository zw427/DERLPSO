import numpy as np
import random
from sde_equations import SDE_Equation
from interface import DE_Models
from Estimators.estimator import Estimator
from parameter import Parameter


class SDE_Models(DE_Models):
    def __init__(self, equation: SDE_Equation):
        self.equation = equation

    def _solve_sde(
        self, params_dict: dict, y0: np.ndarray, t_eval: np.ndarray, seed: int = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        dt = t_eval[1] - t_eval[0]
        n_steps = len(t_eval)
        y0 = np.atleast_1d(y0)
        dim = len(y0)

        x = np.zeros((n_steps, dim))
        x[0, :] = y0
        sqrt_dt = np.sqrt(dt)

        corr_matrix = (
            self.equation.correlation_matrix
            if self.equation.correlation_matrix is not None
            else np.eye(dim)
        )
        chol_matrix = np.linalg.cholesky(corr_matrix)
        drift_func = self.equation.drift_template
        diffusion_func = self.equation.diffusion_template

        if dim == 1:
            for i in range(n_steps - 1):
                dW = np.random.normal(0, sqrt_dt)
                drift = drift_func(x[i, 0], t_eval[i], params_dict)
                diffusion = diffusion_func(x[i, 0], t_eval[i], params_dict)
                x[i + 1, 0] = x[i, 0] + drift * dt + diffusion * dW
        else:
            for i in range(n_steps - 1):
                dW_ind = np.random.normal(0, sqrt_dt, dim)
                dW = chol_matrix @ dW_ind
                drift = drift_func(x[i, :], t_eval[i], params_dict)
                diffusion = np.atleast_2d(
                    diffusion_func(x[i, :], t_eval[i], params_dict)
                )
                if diffusion.shape[0] == 1 or diffusion.shape[1] == 1:
                    diffusion = np.diag(diffusion.flatten())
                x[i + 1, :] = x[i, :] + drift * dt + (diffusion @ dW)

        return x

    def simulate(
        self,
        num_data: int,
        parameter: Parameter,
        t_span: list,
        point: int,
        seed: int = None,
    ) -> dict:
        if seed:
            np.random.seed(seed)
        rng = np.random.default_rng(seed)
        time_points = np.linspace(t_span[0], t_span[1], point)
        data_list, param_list, time_list = [], [], []
        param_names = list(self.equation.param_bounds.keys())
        for _ in range(num_data):
            param_values = parameter.sample(rng=rng).squeeze()
            params_dict = dict(zip(param_names, param_values))
            data = self._solve_sde(
                params_dict, self.equation.initial_conditions, time_points
            )
            data_list.append(data)
            param_list.append(param_values)
            time_list.append(time_points)
        return {
            "data": np.stack(data_list),
            "param": np.stack(param_list),
            "time": np.stack(time_list),
        }

    def evaluate(
        self,
        estimator: Estimator,
        test_set: dict,
        train_set: dict = None,
        seed: int = None,
    ) -> dict:
        data, param, time = test_set["data"], test_set.get("param"), test_set["time"]
        prediction = estimator.predict(data, time)
        error = param - prediction if param is not None else None
        mse = np.zeros(data.shape[0])
        param_names = list(self.equation.param_bounds.keys())
        for i in range(data.shape[0]):
            pred_dict = dict(zip(param_names, prediction[i]))
            prediction_data = self._solve_sde(
                pred_dict, self.equation.initial_conditions, time[i], seed=42
            )
            mse[i] = np.mean((data[i] - prediction_data) ** 2)
        return {**test_set, "prediction": prediction, "error": error, "mse": mse}

    def pprint(self, output: dict, estimator_name: str, file: str = None):
        error_mean = "N/A" if output["error"] is None else output["error"].mean(axis=0)
        error_std = "N/A" if output["error"] is None else output["error"].std(axis=0)
        txt = f"""
        ##################################################
        SDE Equation: {self.equation.name}
        Estimator   : {estimator_name}
        --------------------------------------------------
        Parameter Error Mean: {error_mean}
        Parameter Error Std : {error_std}
        MSE Median          : {np.median(output["mse"])}
        MSE Mean            : {output["mse"].mean()}
        MSE Std             : {output["mse"].std()}
        ##################################################\n\n
        """
        if file:
            with open(file, "a", encoding="utf-8") as f:
                f.write(txt)
        else:
            print(txt)
