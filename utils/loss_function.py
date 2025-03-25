import numpy as np
from typing import Union
from abc import ABC

class LossFunction(ABC):
    def gradient(self,
                 prediction:Union[int, float, np.ndarray],
                 target:Union[int, float, np.ndarray]
                ) -> Union[int, float, np.ndarray]:
        pass
    def loss(self,
             prediciton:Union[int, float, np.ndarray],
             target:Union[int, float, np.ndarray]
            ) -> Union[int, float, np.ndarray]:
        pass

class MeanSquareError(LossFunction):
    def gradient(self, prediction, target) -> Union[int, float, np.ndarray]:
        return prediction - target

    def loss(self, prediciton, target) -> Union[int, float, np.ndarray]:
        return np.mean((prediciton - target)**2)

class BinaryCrossEntropy(LossFunction):
    def gradient(self, prediction, target) -> Union[int, float, np.ndarray]:
        return np.mean(prediction - target, 0)

    def loss(self, prediciton, target) -> Union[int, float, np.ndarray]:
        return np.mean(target * np.log(prediciton)
                       + (1 - target) * np.log(1 - prediciton))
