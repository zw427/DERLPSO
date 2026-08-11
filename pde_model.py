import random
from typing import Union

import numpy as np
from scipy.stats import truncnorm

from pde_equations import PDE_Equation
from interface import DE_Models
from Estimators.estimator import Estimator


class PDE_Models(DE_Models):
    def __init__(self, equation: PDE_Equation):
        """
        :param equation: PDE_Equation instance that defines the PDE system.
        """
        self.equation = equation

    def integrate(self, X: Union[np.ndarray, float]) -> np.ndarray:
        """
        Integrate the ODE system defined by the equation with given parameters.
        """
        # if X is single element np.ndarray, .tolist() makes it a Python float
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(X, float):
            X = [X]
        return self.equation.f()(X)

    def simulate(
        self,
        num_data: int,
        mu: float = 0.5,
        sigma: float = 0.5,
        lower: float = 0.0001,
        upper: float = 1.0,
        seed: int = None,
    ) -> np.ndarray:
        """
        Generate random parameters using truncated normal distribution.
        """
        if seed is not None:
            np.random.seed(seed)

        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        truncated_normal_dist = truncnorm(a, b, loc=mu, scale=sigma)
        return truncated_normal_dist.rvs(num_data, random_state=seed)

    def parameter_est(self, estimator: Estimator, data: np.ndarray) -> np.ndarray:
        """
        Estimate parameters using the provided estimator and data.
        """
        return estimator.predict(data, None)

    def evaluate(self, estimator: Estimator, test_set: dict, seed: int = None) -> dict:
        """
        Evaluate the estimator on the test set and return results.
        """
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        data = test_set.get("data", None)
        assert data is not None
        param = test_set.get("param", None)

        prediction = np.zeros((data.shape[0], self.equation.num_param))
        for i in range(data.shape[0]):
            prediction[i] = self.parameter_est(estimator, data[i])

        if param is None:
            return test_set | {"prediction": prediction}

        # error between parameters
        error = param - prediction

        # mse of each sample
        mse = np.zeros((data.shape[0]))
        for i in range(data.shape[0]):
            prediction_data = self.integrate(prediction[i].squeeze())
            mse[i] = np.mean((data[i] - prediction_data) ** 2)

        # return data, time, param, prediction, error, and mse
        return test_set | {"prediction": prediction, "error": error, "mse": mse}

    def pprint(self, output: dict, estimator_name: str, file: str = None):
        """
        Pretty print (or save) the results of the evaluation.
        """
        txt = f"""
        ##################################################\n
        PDE Equation: {self.equation.name}\n
        Estimator   : {estimator_name}\n
        --------------------------------------------------\n
        Parameter Error Mean: {output["error"].mean(axis=0)}\n
        Parameter Error Std : {output["error"].std(axis=0)}\n
        MSE Median          : {np.median(output["mse"])}\n
        MSE Mean            : {output["mse"].mean()}\n
        MSE Std             : {output["mse"].std()}\n
        ##################################################\n\n\n
        """
        print(txt)
        if file:
            with open(file, "a") as f:
                f.write(txt)
