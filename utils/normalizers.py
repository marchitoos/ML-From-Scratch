import numpy as np
from typing import Union, Tuple
from abc import ABC

class Normalizer(ABC):
    def __call__(self, array:Union[int, float, np.ndarray],
                 axis:Tuple[int, ...]) -> Union[int, float, np.ndarray]:
        pass

class MinMax(Normalizer):
    def __call__(self, array, axis) -> Union[int, float, np.ndarray]:
        mm = np.max(array, axis) - np.min(array, axis, keepdims=True)
        return (array - np.min(array, axis, keepdims=True)) / mm
    
class Mean(Normalizer):
    def __call__(self, array, axis) -> Union[int, float, np.ndarray]:
        mm = np.max(array, axis) - np.min(array, axis, keepdims=True)
        return (array - np.mean(array, axis, keepdims=True)) / mm

class ZScore(Normalizer):
    def __call__(self, array, axis) -> Union[int, float, np.ndarray]:
        mean = np.mean(array, axis, keepdims=True)
        std = np.std(array , axis, keepdims=True)
        return (array - mean)/std
