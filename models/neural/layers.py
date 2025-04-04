import numpy as np
from typing import Type
from abc import ABC, abstractmethod
from utils.functions import Function, Sigmoid, Tanh

class Layer(ABC):
    """An abstract Layer.
    
    Attributes:
        weights (np.ndarray): The weights.
        bias (np.ndarray): The bias.
        z (np.ndarray): Keeps the partially computed output.
        inputs (np.ndarray): Keeps the last input.
        output (np.ndarray): Keeps the last output.
        values (np.ndarray): The Weights and Biases.

    Methods:
        forwardpass(inputs): The forward pass of the regression.
        backwardpass(inputs): The backward pass of the regression.
    """
    @abstractmethod
    def __init__(self, input_size:int, units:int, function:Type[Function]):
        """Constructor
        
        Args:
            input_size (int): The dimensionality of the input data.
            units (int): The number of neurons in the layer.
            function (Type[Function]): The activation function.
        """
        self.weights = np.empty()
        self.bias = np.empty()

        self.function = None

        self.inputs = np.empty()
        self.z = np.empty()
        self.output = np.empty()
    
    @abstractmethod
    def forwardpass(self, inputs:np.ndarray):
        """The Forward Pass.


        Perform the forawrd pass throgh the layer.
        
        Args:
            inputs (np.ndarray): Input data.
        
        Return:
            np.ndarray: The result of the transformations.
        """

    @abstractmethod
    def backwardpass(self, error:np.ndarray, learning_rate:float):
        """The Backward Pass

        
        Perform backward pass using gradient descent.
        
        Args:
            error: Gradient from the error.
            learning_rate: Learning rate for values update.
        
        Return:
            np.ndarray: The gradient of the layer.
        """
    
    @property
    def values(self) -> np.ndarray:
        """The Weights and Biases"""
        return np.concatenate((self.weights.flatten(), self.bias.flatten()), axis=0)

    @values.setter
    def values(self, store:np.ndarray) -> None:
        """Set the Weights and Biases"""
        size = self.weights.size
        self.weights = store[:size].reshape(self.weights.shape)
        self.bias = store[size:].reshape(self.bias.shape)

class Dense(Layer):
    """A Dense Layer.
    
    Attributes:
        weights (np.ndarray): The weights.
        bias (np.ndarray): The bias.
        z (np.ndarray): Keeps the partially computed output.
        inputs (np.ndarray): Keeps the last input.
        output (np.ndarray): Keeps the last output.
        values (np.ndarray): The Weights and Biases.

    Methods:
        forwardpass(inputs): The forward pass of the regression.
        backwardpass(inputs): The backward pass of the regression.
    """
    def __init__(self, input_size:int, units:int, function:Type[Function]) -> None:
        self.weights = np.random.randn(input_size, units)
        self.bias = np.zeros((1, units))

        self.function = function

        self.inputs = np.empty((1, input_size))
        self.z = np.empty((1, units))
        self.output = np.empty((1, units))

    def forwardpass(self, inputs) -> np.ndarray:
        self.inputs = inputs

        self.z = inputs @ self.weights + self.bias
        self.output = self.function(self.z)

        return self.output
    
    def backwardpass(self, error, learning_rate) -> np.ndarray:
        dz = error * self.function(self.z, d=True)
        dw = self.inputs.T @ dz
        dx = dz @ self.weights.T

        self.weights -= dw * learning_rate
        self.bias -= dz * learning_rate
        return dx

class Recurrent(Layer):
    """A Recurrent Layer.
    
    Attributes:
        weights (np.ndarray): The weights.
        hweights (np.ndarray): The hidden weights.
        bias (np.ndarray): The bias.
        hz (np.ndarray): Keeps the hidden state.
        z (np.ndarray): Keeps the partially computed output.
        inputs (np.ndarray): Keeps the last input.
        output (np.ndarray): Keeps the last output.
        values (np.ndarray): The Weights and Biases.

    Methods:
        forwardpass(inputs): The forward pass of the regression.
        backwardpass(inputs): The backward pass of the regression.
    """
    def __init__(self, input_size:int, units:int, function:Type[Function]=Tanh) -> None:
        self.weights = np.random.randn(input_size, units)
        self.hweights = np.random.randn(1, units)
        self.bias = np.zeros((1, units))

        self.function = function

        self.inputs = np.empty((1, input_size))
        self.h = np.zeros((1, units))
        self.z = np.empty((1, units))
        self.output = np.empty((1, units))
    
    def forwardpass(self, inputs) -> np.ndarray:
        self.inputs = inputs
        self.h = self.output

        self.z = inputs @ self.weights + self.h * self.hweights + self.bias
        self.output = self.function(self.z)

        return self.output
    
    def backwardpass(self, error, learning_rate) -> np.ndarray:
        dz = error * self.function(self.z, d=True)
        dw = self.inputs.T @ dz
        dh = self.h * dz
        dx = dz @ self.weights.T

        self.weights -= dw * learning_rate
        self.hweights -= dh * learning_rate
        self.bias -= dz * learning_rate
        return dx
