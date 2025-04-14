import numpy as np
from models.trees.nodes import Node, DecisionNode
from utils.other_utils import mode, gini
from typing import Type

class RandomFlorest:
    def __init__(self, n_trees, depth, samples, impurity_function=gini, node_type:Type[Node]=DecisionNode) -> None:
        self.n_trees = n_trees
        self.trees = [node_type(depth, samples, impurity_function) for _ in range(n_trees)]
        self.is_leaf = False
        self.value = None
        self.children = None

    def forwardpass(self, value):
        predictions = np.array([tree.forwardpass(value) for tree in self.trees])
        return mode(predictions)
