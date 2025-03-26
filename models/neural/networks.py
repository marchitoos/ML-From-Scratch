import numpy as np
from typing import Type
from models.neural.layers import Layer

class Network:
    def __init__(self, *args:Type[Layer]):
        self.layers = list(args)
    
    def add_layer(self, *args):
        self.layers.extend(args)
    
    def forwardpass(self, inputs):
        for layer in self.layers:
            inputs = layer.forwardpass(inputs)
        return inputs
    
    def backwardpass(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = layer.backwardpass(error, learning_rate)
