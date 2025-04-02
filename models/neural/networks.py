import numpy as np
from typing import Type, Tuple
from models.neural.layers import Layer
from utils.loss_function import LossFunction
from utils.learning_schedules import LearningSchedule, Constant

class Network:
    def __init__(self, *args:Type[Layer]):
        self.layers = list(args)
        self.loss_function = None
        self.learning_schedule = None
        self._configd = False
    
    def add_layer(self, *args):
        self.layers.extend(args)
    
    def forwardpass(self, inputs):
        for layer in self.layers:
            inputs = layer.forwardpass(inputs)
        return inputs
    
    def backwardpass(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = layer.backwardpass(error, learning_rate)
    
    def config(self, loss_function:Type[LossFunction],
               learning_schedule:Type[LearningSchedule]=Constant(0.01)):
        self.loss_function = loss_function
        self.learning_schedule = learning_schedule
        self._configd = True

    def train(self, X, Y, epochs:int) -> Tuple[float]:
        if not self._configd:
            raise ValueError("Network not configured. Call config() before training.")
        loss = []
        for epoch in range(1, epochs + 1):
            learning_rate = self.learning_schedule.learning_rate(epoch)
            for i in range(len(X)):
                r = self.forwardpass(X[i:i+1])
                error = self.loss_function.gradient(r, Y[i:i+1])
                loss.append(self.loss_function.loss(r, Y[i:i+1]))
                self.backwardpass(error, learning_rate)
            
            if (epoch % (epochs // 10) == 0) or (epoch in {epochs, 1}):
                print(f"Epoch {epoch}/{epochs}, Loss: {np.mean(loss[-1])}")

        return loss
