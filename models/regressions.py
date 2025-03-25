import numpy as np
from utils.functions import Sigmoid

class LinearRegression:
    def __init__(self, input_size:int) -> None:
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1, 1))

        self.inputs = np.empty((1, input_size))
        self.output = np.empty((1, 1))
    
    def forwardpass(self, inputs:np.ndarray) -> np.ndarray:
        self.inputs = inputs
        
        self.output = inputs @ self.weights + self.bias

        return self.output
    
    def backwardpass(self, error:np.ndarray, learning_rate:float) -> np.ndarray:
        dw = self.inputs.T @ error

        self.bias -= error * learning_rate
        self.weights -= dw * learning_rate

class LogisticRegression:
    def __init__(self, input_size:int, sigmoid_value:float=1) -> None:
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1, 1))

        self.function = Sigmoid(sigmoid_value)

        self.inputs = np.empty((1, input_size))
        self.z = np.empty((1, 1))
        self.output = np.empty((1, 1))

    def forwardpass(self, inputs:np.ndarray) -> np.ndarray:
        self.inputs = inputs
        
        self.z = inputs @ self.weights + self.bias
        self.output = self.function(self.z)

        return self.output

    def backwardpass(self, error:np.ndarray, learning_rate:float) -> np.ndarray:
        dw = self.inputs.T @ error

        self.bias -= error * learning_rate
        self.weights -= dw * learning_rate
