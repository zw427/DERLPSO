import os
from typing import List

import numpy as np
import torch
from scipy.integrate import solve_ivp

from equation import Equation, ODE_Equation, PDE_Equation
from parameter import Parameter
from Estimators.estimator import Estimator


class DE_Models: 
    def __init__():
        raise NotImplementedError()

    def integrate():
        raise NotImplementedError() 

    def simulate():
        raise NotImplementedError()

    def parameter_est():
        raise NotImplementedError() 

    def evaluate():
        raise NotImplementedError() 


class ODE_Models(DE_Models):
    def __init__(self, equation: ODE_Equation):
        self.equation = equation

    def integrate(self, args: np.ndarray, y0: np.ndarray, 
                  t_eval: np.ndarray, method: str ='RK45'):
        return solve_ivp(self.equation.f(), [t_eval[0], t_eval[-1]], y0, 
                         method, t_eval=t_eval, args=(args))

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

            if not data.success:
                continue
            else:
                data = np.transpose(data.y, (1, 0))

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

        predicted = np.zeros((data.shape[0], self.equation.num_param))

        for i in range(data.shape[0]):
            predicted[i] = self.parameter_est(estimator, data[i], time[i])

        if param is None:
            return {'data': data, 'time': time, 'predicted': predicted}

        error = param - predicted
        
        mse = np.zeros((data.shape[0]))
        for i in range(predicted.shape[0]):
            predicted_data = self.integrate(torch.as_tensor(predicted[i])[None], data[i, 0], time[i])
            mse[i] = np.mean((data[i] - predicted_data.y.T) ** 2)
        
        return {'data': data, 'time': time, 'param': param, 
                'predicted': predicted, 'error': error, 'mse': mse}
    