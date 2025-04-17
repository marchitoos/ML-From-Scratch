import numpy as np
from typing import Union
from abc import ABC, abstractmethod

class Function(ABC):
    """An abstract Function"""
    @abstractmethod
    def __call__(self, x:Union[int, float, np.ndarray], d:bool=False):
        """Call the function.
        
        Args:
            x (int | float | np.ndarray): The x.
            d (bool), optional: If is the derivative or not.
        
        Returns:
            (int | float | np.ndarray): The f(x).
        """
        pass

class Sigmoid(Function):
    """A Sigmoid Function."""
    def __init__(self, k:float=1):
        """Constructor
        
        Args:
            k (float): The multiplier of x in e^-(k * x).
        """
        self.k = k

    def __call__(self, x, d=False) -> Union[int, float, np.ndarray]:
        a = 1 / (1 + np.exp(self.k * -x))
        if d:
            return self.k * a * (1 - a)
        return a

class Tanh(Function):
    """A TanH Function."""
    def __call__(self, x, d=False) -> Union[int, float, np.ndarray]:
        a = np.tanh(x)
        if d:
            return 1 - a**2
        return a

class ReLU(Function):
    """A ReLU Function."""
    def __call__(self, x, d=False) -> Union[int, float, np.ndarray]:
        if d:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

class LeakyReLU(Function):
    """A Leaky ReLU Function."""
    def __init__(self, alpha:float=0.01):
        """Constructor
        
        Args:
            alpha (float): The slope of the negative part.
        """
        self.alpha = alpha

    def __call__(self, x, d=False) -> Union[int, float, np.ndarray]:
        if d:
            return np.where(x > 0, 1, self.alpha)
        return np.where(x > 0, x, self.alpha * x)

class Softmax(Function):
    """A Softmax Function."""
    def __call__(self, x, d=False) -> Union[int, float, np.ndarray]:
        x = np.clip(x, -700, 700)
        a = np.exp(x)
        s = a/np.sum(a, axis=1)
        if d:
            s = s.reshape(-1, 1)
            return np.diagflat(s) - (s @ s.T)
        return s
