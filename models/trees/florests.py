import numpy as np
from models.trees.nodes import Node, DecisionNode
from utils.other_utils import mode, gini
from typing import Type

class RandomFlorest:
    """A Random Florest.
    
    Attributes:
        n_trees (int): The number of trees in the florest.
        trees (list): The trees in the florest.
        is_leaf (bool): Whether the node is a leaf or not.
        value (float): The value of the node.
        children (list): The children of the node.
    
    Methods:
        forwardpass(value): The forward pass of the florest.
    """
    def __init__(self, n_trees, depth, samples, impurity_function=gini, node_type:Type[Node]=DecisionNode) -> None:
        self.n_trees = n_trees
        self.trees = [node_type(depth, samples, impurity_function) for _ in range(n_trees)]
        self.is_leaf = False
        self.value = None
        self.children = None

    def forwardpass(self, value):
        """The Forward Pass.

        Args:
            value (np.ndarray): Input data.
        
        Returns:
            np.ndarray: The result of the transformations.
        """
        predictions = np.array([tree.forwardpass(value) for tree in self.trees])
        return mode(predictions, axis=0).reshape(-1, 1)
