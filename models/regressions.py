import numpy as np
from utils.functions import Sigmoid
from abc import ABC, abstractmethod

class Regression(ABC):
    """An abstract Regression.
    
    Attributes:
        weights (np.ndarray): The weights.
        bias (np.ndarray): The bias.
        inputs (np.ndarray): Keeps the last input.
        output (np.ndarray): Keeps the last output.

    Methods:
        forwardpass(inputs): The forward pass of the regression.
        backwardpass(inputs): The backward pass of the regression.
    """
    def __init__(self, input_size:int) -> None:
        """Constructor
        
        Args:
            input_size (int): The dimensionality of the input data.
        """
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1, 1))

        self.inputs = np.empty((1, input_size))
        self.output = np.empty((1, 1))
    
    @abstractmethod
    def forwardpass(self, inputs:np.ndarray):
        """The Forward Pass.


        Perform the forawrd pass throgh the layer.
        
        Args:
            inputs (np.ndarray): Input data.
        
        Return:
            np.ndarray: The result of the transformations.
        """
        raise NotImplementedError

    @abstractmethod
    def backwardpass(self, error:np.ndarray, learning_rate:float):
        """The Backward Pass

        
        Perform backward pass using gradient descent.
        
        Args:
            error: Gradient from the error.
            learning_rate: Learning rate for values update.
        """

class LinearRegression(Regression):
    """A Linear Regression.
    
    Attributes:
        weights (np.ndarray): The weights.
        bias (np.ndarray): The bias.
        inputs (np.ndarray): Keeps the last input.
        output (np.ndarray): Keeps the last output.

    Methods:
        forwardpass(inputs): The forward pass of the regression.
        backwardpass(inputs): The backward pass of the regression.
    """
    def __init__(self, input_size) -> None:
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1, 1))

        self.inputs = np.empty((1, input_size))
        self.output = np.empty((1, 1))
    
    def forwardpass(self, inputs):
        self.inputs = inputs
        
        self.output = inputs @ self.weights + self.bias

        return self.output
    
    def backwardpass(self, error, learning_rate) -> None:
        dw = self.inputs.T @ error

        self.bias -= error * learning_rate
        self.weights -= dw * learning_rate

class LogisticRegression(Regression):
    """A Logistic Regression.
    
    Attributes:
        weights (np.ndarray): The weights.
        bias (np.ndarray): The bias.
        z (np.ndarray): Keeps the partially computed output.
        inputs (np.ndarray): Keeps the last input.
        output (np.ndarray): Keeps the last output.

    Methods:
        forwardpass(inputs): The forward pass of the regression.
        backwardpass(inputs): The backward pass of the regression.
    """
    def __init__(self, input_size, sigmoid_value:float=1) -> None:
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1, 1))

        self.function = Sigmoid(sigmoid_value)

        self.inputs = np.empty((1, input_size))
        self.z = np.empty((1, 1))
        self.output = np.empty((1, 1))

    def forwardpass(self, inputs) -> np.ndarray:
        self.inputs = inputs
        
        self.z = inputs @ self.weights + self.bias
        self.output = self.function(self.z)

        return self.output

    def backwardpass(self, error, learning_rate) -> None:
        dw = self.inputs.T @ error

        self.bias -= error * learning_rate
        self.weights -= dw * learning_rate
