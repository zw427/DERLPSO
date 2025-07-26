import os
from typing import List

import numpy as np
import torch

from data import generate_data, read_data
from equations import ODE_Equation
from models.base_model import BaseModel
from models.ml_models import MLModel

class ODE_Models:
    def __init__(self, equation: ODE_Equation, init_data: List[float], 
                 param_mu: List[float], param_sigma: List[float]):
        assert len(param_mu) == len(param_sigma), "param_mu and param_sigma must have the same length"
        self.equation = equation
        self.init_data = init_data
        self.param_mu = param_mu
        self.param_sigma = param_sigma


    def get_filenames(self, interval: List[float], point: int, train: bool) -> List[str]:
        mode = 'train' if train else 'test'
        data_file =  f'data/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}/{mode}/data.csv'
        time_file =  f'data/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}/{mode}/time.csv'
        param_file = f'data/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}/{mode}/param.csv'
        model_dir = f'saved_models/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}'
        return data_file, time_file, param_file, model_dir
    

    def simulate_data(self, num_data: int, interval: List[float], points: List[int], seed: int = None, train: bool = True, over_write: bool = False) -> None:    
        
        assert len(interval) == 2, "interval must have exactly two floats [start, end]"
        
        if not seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print("Seed set to:", seed)

        for point in points:
            data_file, time_file, param_file, _ = self.get_filenames(interval, point, train)
            if not (os.path.exists(data_file) and os.path.exists(time_file) and os.path.exists(param_file)) or over_write:
                generate_data(
                    equation=self.equation,
                    param_mu=self.param_mu,
                    param_sigma=self.param_sigma,
                    init_data=self.init_data,
                    point=point,
                    interval=interval,
                    num_data=num_data,
                    data_filename=data_file,
                    time_filename=time_file,
                    param_filename=param_file
                )



    def evaluate(self, models: List[BaseModel], interval: List[float], points: List[int], seed: int = None) -> None:
        
        assert len(interval) == 2, "interval must have exactly two floats [start, end]"
        
        if not seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print("Seed set to:", seed)

        if any([isinstance(model, MLModel) for model in models]):
            for point in points:
                data_file, time_file, param_file, _ = self.get_filenames(interval, point, train=True)
                if not (os.path.isfile(data_file) and os.path.isfile(time_file) and os.path.isfile(param_file)):
                    print("Please simulate the training data first.")
                    return 

        for point in points:
            data_file, time_file, param_file, _ = self.get_filenames(interval, point, train=False)
            if not (os.path.isfile(data_file) and os.path.isfile(time_file) and os.path.isfile(param_file)):
                print("Please simulate the testing data first.")
                return 

        ### evaluate here
