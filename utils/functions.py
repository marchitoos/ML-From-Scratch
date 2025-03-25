import numpy as np
from typing import Union
from abc import ABC

class Function(ABC):
    def __call__(self, x:Union[int, float, np.ndarray], d:bool=True):
        pass

class Sigmoid(Function):
    def __init__(self, k:float=1):
        self.k = k

    def __call__(self, x, d=False) -> Union[int, float, np.ndarray]:
        a = 1 / (1 + np.exp(self.k * -x))
        if d:
            return self.k * a * (1 - a)
        return a

class Tanh(Function):
    def __call__(self, x, d=False) -> Union[int, float, np.ndarray]:
        a = np.tanh(x)
        if d:
            return 1 - a**2
        return a
