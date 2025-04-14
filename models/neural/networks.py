import numpy as np
from typing import Type, List
from models.neural.layers import Layer
from utils.loss_functions import LossFunction
from utils.learning_schedules import LearningSchedule, Constant

class Network:
    """A Neural Network.
    
    Attributes:
        layers (list): The layers of the network.
        loss_function (LossFunction): The loss function.
        learning_schedule (LearningSchedule): The learning schedule.
        _configd (bool): Whether the network is configured or not.
    
    Methods:
        add_layer(*args): Add layers to the network.
        forwardpass(inputs): The forward pass of the network.
        backwardpass(error, learning_rate): The backward pass of the network.
        config(loss_function, learning_schedule): Configure the network.
        train(X, Y, epochs): Train the network.
        store(): Store the weights and biases.
        load(store): Load the weights and biases.
    """
    def __init__(self, *args:Type[Layer]):
        """Constructor
        
        Args:
            *args (Type[Layer]): The layers of the network.
        """
        self.layers = list(args)
        self.loss_function = None
        self.learning_schedule = None
        self._configd = False
    
    def add_layer(self, *args):
        """Add layers to the network.
        
        Args:
            *args (Type[Layer]): The layers of the network.
        """
        self.layers.extend(args)
    
    def forwardpass(self, inputs):
        """The Forward Pass.


        Perform the forawrd pass throgh the regression.
        
        Args:
            inputs (np.ndarray): Input data.
        
        Return:
            np.ndarray: The result of the transformations.
        """
        for layer in self.layers:
            inputs = layer.forwardpass(inputs)
        return inputs
    
    def backwardpass(self, error, learning_rate):
        """The Backward Pass

        
        Perform backward pass using gradient descent.
        
        Args:
            error: Gradient from the error.
            learning_rate: Learning rate for values update.
        """
        for layer in reversed(self.layers):
            error = layer.backwardpass(error, learning_rate)
    
    def config(self, loss_function:Type[LossFunction],
               learning_schedule:Type[LearningSchedule]=Constant(0.01)):
        """Configure the network.
        
        Args:
            loss_function (Type[LossFunction]): The loss function.
            learning_schedule (Type[LearningSchedule]): The learning schedule.
        """
        self.loss_function = loss_function
        self.learning_schedule = learning_schedule
        self._configd = True

    def train(self, X, Y, epochs:int) -> List[float]:
        """Train the network.
        
        Args:
            X (np.ndarray): The input data.
            Y (np.ndarray): The target data.
            epochs (int): The number of epochs.
        
        Returns:
            List[float]: The loss of the network.
        """
        if not self._configd:
            raise ValueError("Network not configured. Call config() before training.")
        loss = []
        for epoch in range(1, epochs + 1):
            learning_rate = self.learning_schedule.learning_rate(epoch)
            ls = 0
            for i in range(len(X)):
                r = self.forwardpass(X[i:i+1])
                error = self.loss_function.gradient(r, Y[i:i+1])
                ls += self.loss_function.loss(r, Y[i:i+1])
                self.backwardpass(error, learning_rate)
            loss.append(ls / len(X))
            
            if (epoch % (epochs // 10) == 0) or (epoch in {epochs, 1}):
                print(f"Epoch {epoch}/{epochs}, Loss: {np.mean(loss[-1])}")

        return loss

    def store(self):
        """Store the weights and biases"""
        return [layer.values for layer in self.layers]
    
    def load(self, store):
        """Load the weights and biases"""
        for i, layer in enumerate(self.layers):
            layer.values = store[i]
