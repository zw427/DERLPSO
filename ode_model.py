import os
from typing import List

import numpy as np
import torch
from scipy.integrate import solve_ivp, odeint

from equation import Equation, ODE_Equation, PDE_Equation
from parameter import Parameter
from Estimators.estimator import Estimator
# from Estimators.ml_estimator import MLEstimator

from interface import DE_Models


class ODE_Models(DE_Models):
    def __init__(self, equation: ODE_Equation):
        self.equation = equation

    def integrate(self, args: np.ndarray, y0: np.ndarray, 
                  t_eval: np.ndarray):
        return odeint(self.equation.f(), y0, t_eval, args=(args.squeeze(),), tfirst=True)
        

    def simulate(self, num_data: int, parameter: Parameter, init_data: List[float], 
                      interval: List[float], point: int, seed: int = None) -> torch.Tensor:    
        
        assert len(interval) == 2, "interval must have exactly two floats [start, end]"
        
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print("Seed set to:", seed)

        time_points = np.linspace(interval[0], interval[-1], point)
        data_list, param_list, time_list  = [], [], []

        while len(data_list) < num_data:

            args = parameter.sample()
            args = np.abs(args)
            y0 = np.array(init_data)

            data = self.integrate(args, y0, time_points)

            # if not data.success:
            #     continue
            # else:
            # data = np.transpose(data, (1, 0))

            data_list.append(data)
            param_list.append(args)
            time_list.append(time_points)

        # data.shape = [num_data, point, num_state]
        data = np.stack(data_list, axis=0)
        
        # param.shape = [num_data, num_param]
        param = np.concatenate(param_list, axis=0)

        # time.shape = [num_data, point]
        time = np.stack(time_list, axis=0)

        return {'data': data, 'param': param, 'time': time}

    def parameter_est(self, estimator: Estimator, data: torch.Tensor, time: torch.Tensor):
        return estimator.predict(data, time)

    def evaluate(self, estimator: Estimator, data: torch.Tensor, time: torch.Tensor, 
                 param: torch.Tensor = None, seed: int= None):
                 
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print("Seed set to:", seed)

        # if isinstance(estimator, MLEstimator):
        #     predicted = self.parameter_est(estimator, data, time)
        # else:
        predicted = np.zeros((data.shape[0], self.equation.num_param))
        fit = np.zeros((data.shape[0]))
        for i in range(data.shape[0]):
            predicted[i] = estimator.predict(data[i], time[i])

            
        if param is None:
            return {'data': data, 'time': time, 'predicted': predicted}

        error = param - predicted
        
        mse = np.zeros((data.shape[0]))
        for i in range(predicted.shape[0]):
            predicted_data = self.integrate(torch.as_tensor(predicted[i])[None], data[i, 0], time[i])
            mse[i] = np.mean((data[i] - predicted_data) ** 2)
        
        return {'data': data, 'time': time, 'param': param, 
                'predicted': predicted, 'error': error, 'mse': mse}
    