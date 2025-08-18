from typing import List

import numpy as np
import scipy

from ode_equations import ODE_Equation
from parameter import Parameter
from interface import DE_Models
from Estimators.estimator import Estimator
from Estimators.ml_estimator import MLEstimator


class ODE_Models(DE_Models):

    def __init__(self, equation: ODE_Equation):
        '''
        :param equation: ODE_Equation instance that defines the ODE system.
        '''
        self.equation = equation

    def integrate(self, args: np.ndarray, y0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        '''
        Integrate the ODE system defined by the equation with given parameters.
        '''
        return scipy.integrate.odeint(self.equation.f(), y0, t_eval, 
                                      args=(args,), tfirst=self.equation.t_first)
        
    def simulate(self, num_data: int, parameter: Parameter, init_data: List[float], 
                      interval: List[float], point: int, seed: int = None) -> np.ndarray:    
        '''
        Simulate the ODE system and generate num_data of samples.
        '''
        assert len(interval) == 2, "interval must have exactly two floats [start, end]"
        
        # set seed if any
        np.random.seed(seed) if seed else None

        # prepare inputs to integrate
        time_points = np.linspace(interval[0], interval[1], point)
        data_list, param_list, time_list = [], [], []

        while len(data_list) < num_data:
            try:
                args = parameter.sample()
                args = np.abs(args.squeeze())
                y0 = np.array(init_data)

                data = self.integrate(args, y0, time_points)

                data_list.append(data)
                param_list.append(args)
                time_list.append(time_points)
            except:
                continue

        # data.shape = [num_data, point, num_state]
        data = np.stack(data_list, axis=0)

        # param.shape = [num_data, num_param]
        param = np.stack(param_list, axis=0)

        # time.shape = [num_data, point]
        time = np.stack(time_list, axis=0)

        return {'data': data, 'param': param, 'time': time}

    def parameter_est(self, estimator: Estimator, data: np.ndarray, time: np.ndarray) -> np.ndarray:
        '''
        Estimate parameters using the provided estimator and data.
        '''
        return estimator.predict(data, time)

    def evaluate(self, estimator: Estimator, test_set: dict, train_set: dict = None, 
                 seed: int = None) -> dict:
        '''
        Evaluate the estimator on the test set and return results.
        '''
        # assert 'data' and 'time' in test_set
        assert set(['data', 'time']).issubset(set(test_set.keys()))
        data = test_set.get('data', None)
        param = test_set.get('param', None)
        time = test_set.get('time', None)

        # assert 'data', 'param' and 'time' in train_set if not None
        assert set(['data', 'time', 'param']).issubset(set(train_set.keys())) if train_set else None

        # set seed if any
        np.random.seed(seed) if seed else None

        # if ML Estimator, then train on train_set (can batch predict)
        if isinstance(estimator, MLEstimator):
            assert train_set is not None
            # estimator.train(train_set, seed) # TODO
            prediction = self.parameter_est(estimator, data, time)
        else:
            # get predictions from estimator
            prediction = np.zeros((data.shape[0], self.equation.num_param))
            for i in range(data.shape[0]):
                prediction[i] = self.parameter_est(estimator, data[i], time[i])

        # if no parameter is provided, return data, time and prediction only
        if param is None:
            return test_set | {'prediction': prediction}

        # error between parameters
        error = param - prediction
        
        # mse of each sample
        mse = np.zeros((data.shape[0]))
        for i in range(data.shape[0]):
            prediction_data = self.integrate(prediction[i].squeeze(), data[i, 0], time[i])
            mse[i] = np.mean((data[i] - prediction_data) ** 2)
        
        # return data, time, param, prediction, error, and mse
        return test_set | {'prediction': prediction, 'error': error, 'mse': mse}
    
    def pprint(self, output: dict, file: str = None):
        '''
        Pretty print the results of the evaluation.
        '''
        print(output)
