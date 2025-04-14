import numpy as np
from utils.other_utils import mode, gini, variance
import operator as op
from abc import ABC

class Node(ABC):
    def __init__(self, depth, samples, impurity_function=gini, leaf_function=mode) -> None:
        if depth == 0:
            self.is_leaf = True
            self.value = leaf_function(samples[:, -1])
            self.children = None
            return

        X = samples[:, : -1]
        Y = samples[:, -1]

        self.impurity = impurity_function(Y)
        self.condition = None
        best_splitX = None
        best_splitY = None

        for i in range(X.shape[1]):
            options = np.unique(X[:, i])
            conditions = [op.eq, op.gt, op.lt]
            for condition in conditions:
                for option in options:
                    split = condition(X[:, i], option)
                    if np.any(split) and np.any(~split):
                        splitedX = [X[split], X[~split]]
                        splitedY = [Y[split], Y[~split]]
                        impurity = (impurity_function(splitedY[0])
                                   +impurity_function(splitedY[1]))
        
                        if impurity < self.impurity:
                            self.condition = (condition, i, option)
                            self.impurity = impurity
                            best_splitX = splitedX
                            best_splitY = splitedY
        
        if self.condition is None:
            self.is_leaf = True
            self.value = leaf_function(samples[:, -1])
            self.children = None
            return
        
        self.is_leaf = False
        self.value = None
        best_split = [np.column_stack((best_splitX[0], best_splitY[0])),
                      np.column_stack((best_splitX[1], best_splitY[1]))]
        self.children = [self.__class__(depth-1, best_split[0], impurity_function),
                         self.__class__(depth-1, best_split[1], impurity_function)]

    def forwardpass(self, value):
        if self.is_leaf:
            return self.value

        if self.condition[0](value[:, self.condition[1]], self.condition[2]):
            return self.children[0].forwardpass(value)
        
        return self.children[1].forwardpass(value)

class DecisionNode(Node):
    def __init__(self, depth, samples, impurity_function=gini) -> None:
        super().__init__(depth, samples, impurity_function, leaf_function=mode)

class RegressionNode(Node):
    def __init__(self, depth, samples, impurity_function=variance) -> None:
        super().__init__(depth, samples, impurity_function, leaf_function=np.mean)
