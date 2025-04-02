import numpy as np
from typing import Union
from abc import ABC, abstractmethod

class LearningSchedule(ABC):
    """An abstract Learning Schedule

    Methods:
        learning_rate(epoch): The learning rate.
    """
    @abstractmethod
    def learning_rate(self, epoch:int) -> Union[float]:
        """The Learning Rate

        The learning rate.

        Args:
            epoch (int): The current epoch.

        Returns:
            (float): The learning rate.
        """

class Constant(LearningSchedule):
    """A Constant Learning Rate

    Methods:
        learning_rate(epoch): The learning rate.
    """
    def __init__(self, learning_rate:float) -> None:
        self.initial_learning_rate = learning_rate

    def learning_rate(self, epoch:int) -> Union[float]:
        return self.initial_learning_rate

class ExponentialDecay(LearningSchedule):
    """An Exponential Decay Learning Rate

    Methods:
        learning_rate(epoch): The learning rate.
    """
    def __init__(self, initial_learning_rate:float, decay_rate:float) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def learning_rate(self, epoch:int) -> Union[float]:
        return self.initial_learning_rate * np.exp(-self.decay_rate * epoch)

class TimeDecay(LearningSchedule):
    """A Time Based Decay Learning Rate

    Methods:
        learning_rate(epoch): The learning rate.
    """
    def __init__(self, initial_learning_rate:float, decay_rate:float) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def learning_rate(self, epoch:int) -> Union[float]:
        return self.initial_learning_rate / (1 + self.decay_rate * epoch)

class StepDecay(LearningSchedule):
    """A Step Decay Learning Rate

    Methods:
        learning_rate(epoch): The learning rate.
    """
    def __init__(self, initial_learning_rate:float, drop:float, epochs_drop:int) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.drop = drop
        self.epochs_drop = epochs_drop

    def learning_rate(self, epoch:int) -> Union[float]:
        return self.initial_learning_rate * (self.drop ** np.floor(epoch / self.epochs_drop))
