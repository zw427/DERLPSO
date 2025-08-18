from typing import Callable, List
from equation import Equation


class PDE_Equation(Equation):
    def __init__(self, name: str, func: Callable[..., List[float]], num_param: int):
        super().__init__(name, 'PDE', func, num_param)
