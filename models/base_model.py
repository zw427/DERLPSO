
from typing import List, Callable
from equations import Equation

class BaseModel:
    def __init__(self, equation: Equation, init_data: List[float], 
                 param_mu: List[float], param_sigma: List[float]):
        assert len(param_mu) == len(param_sigma), "param_mu and param_sigma must have the same length"
        self.equation = equation
        self.init_data = init_data
        self.param_mu = param_mu
        self.param_sigma = param_sigma


class MLModel(BaseModel):
    def __init__(self, equation: Equation, init_data: List[float], 
                 param_mu: List[float], param_sigma: List[float]):
        super().__init__(equation, init_data, param_mu, param_sigma)
        self.model = None  # Placeholder for the ML model
        self.optimizer = None  # Placeholder for the optimizer
        self.loss_function = None  # Placeholder for the loss function