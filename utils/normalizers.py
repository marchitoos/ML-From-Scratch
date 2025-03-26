import numpy as np
from typing import Union, Tuple
from abc import ABC, abstractmethod

class Normalizer(ABC):
    """An abstract Normalizer"""
    @abstractmethod
    def __call__(self, array:Union[int, float, np.ndarray],
                 axis:Union[int, Tuple[int, ...]]) -> Union[int, float, np.ndarray]:
        """Normalize the data
        
        Args:
            array (int | float | np.ndarray): The data to be normalized.
            axis (int | Tuple[int, ...]): The axis to be normalized.
        
        Returns:
            (int | float | np.ndarray): The normalized data.
        """

class MinMax(Normalizer):
    """A Mininum and Maximum Normalizer"""
    def __call__(self, array, axis=0) -> Union[int, float, np.ndarray]:
        mm = np.max(array, axis) - np.min(array, axis, keepdims=True)
        return (array - np.min(array, axis, keepdims=True)) / mm
    
class Mean(Normalizer):
    """A Mean Normalizer"""
    def __call__(self, array, axis=0) -> Union[int, float, np.ndarray]:
        mm = np.max(array, axis) - np.min(array, axis, keepdims=True)
        return (array - np.mean(array, axis, keepdims=True)) / mm

class ZScore(Normalizer):
    """A Z-Score Normalizer"""
    def __call__(self, array, axis=0) -> Union[int, float, np.ndarray]:
        mean = np.mean(array, axis, keepdims=True)
        std = np.std(array , axis, keepdims=True)
        return (array - mean)/std
