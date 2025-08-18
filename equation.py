from typing import Callable, List

class Equation:
    '''
    Base class for differential equations.
    '''
    def __init__(self, name: str, de_type: str, func: Callable[..., List[float]], num_param: int):
        self.name = name
        self.de_type = de_type.upper()
        self.func = func
        self.num_param = num_param

    def f(self) -> Callable[..., List[float]]:
        return self.func
