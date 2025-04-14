import numpy as np
from typing import Union
from abc import ABC, abstractmethod

class LossFunction(ABC):
    """An abstract Loss Function
    
    Methods:
        loss(predition, target): The loss function.
        gradient(prediction, target): The gradient descent of the loss function.
    """
    @abstractmethod
    def loss(self,
             prediciton:Union[int, float, np.ndarray],
             target:Union[int, float, np.ndarray]
            ) -> Union[int, float, np.ndarray]:
        """The Loss Function
        

        The loss function.

        Args:
            predition (int | float | np.ndarray): The predicted value.
            target (int | float | np.ndarray): The actual, and targeted, value.
        
        Returns:
            (int | float | np.ndarray): The loss result.
        """

    @abstractmethod
    def gradient(self,
                 prediction:Union[int, float, np.ndarray],
                 target:Union[int, float, np.ndarray]
                ) -> Union[int, float, np.ndarray]:
        """The Gradient Descent
        

        The gradient descent of the loss function.

        Args:
            predition (int | float | np.ndarray): The predicted value.
            target (int | float | np.ndarray): The actual, and targeted, value.
        
        Returns:
            (int | float | np.ndarray): The gradient.
        """

class MeanSquareError(LossFunction):
    """A Mean Square Error Loss Function
    
    Methods:
        loss(predition, target): The loss function.
        gradient(prediction, target): The gradient descent of the loss function.
    """
    def gradient(self, prediction, target) -> Union[int, float, np.ndarray]:
        return prediction - target

    def loss(self, prediciton, target) -> Union[int, float, np.ndarray]:
        return np.mean((prediciton - target)**2)

class BinaryCrossEntropy(LossFunction):
    """A Binary Cross Entropy Loss Function
    
    Methods:
        loss(predition, target): The loss function.
        gradient(prediction, target): The gradient descent of the loss function.
    """
    def gradient(self, prediction, target) -> Union[int, float, np.ndarray]:
        return np.mean(prediction - target, 0)

    def loss(self, prediciton, target) -> Union[int, float, np.ndarray]:
        prediciton = np.clip(prediciton, 1e-15, 1 - 1e-15)
        return np.mean(target * np.log(prediciton)
                       + (1 - target) * np.log(1 - prediciton))
