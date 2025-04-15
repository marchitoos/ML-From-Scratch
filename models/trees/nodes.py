import numpy as np
from utils.other_utils import mode, gini
import operator as op
from abc import ABC

class Node(ABC):
    """A Node in a decision tree.
    
    Attributes:
        is_leaf (bool): Whether the node is a leaf or not.
        value (float): The value of the node.
        children (list): The children of the node.
        impurity (float): The impurity of the node.
        condition (tuple): The condition of the node.
    
    Methods:
        forwardpass(value): The forward pass of the node.
    """
    def __init__(self, depth:int, samples:np.ndarray, impurity_function=gini, leaf_function=mode) -> None:
        """Contructor
        
        Args:
            depth (int): The depth of the node.
            samples (np.ndarray): The samples of the node.
            impurity_function (function): The impurity function.
            leaf_function (function): The leaf function.
        """
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

    def forwardpass(self, inputs) -> np.ndarray:
        """The Forward Pass
        
        Perform the forward pass through the node.

        Args:
            inputs (np.ndarray): Input data.
        
        Output:
            np.ndarray: The result of the transformations.
        """
        if self.is_leaf:
            return self.value

        out = []
        for i in range(inputs.shape[0]):
            value = inputs[i:i+1]

            if self.condition[0](value[:, self.condition[1]], self.condition[2]):
                out.append(self.children[0].forwardpass(value).reshape(-1))
            else:
                out.append(self.children[1].forwardpass(value).reshape(-1))
        
        return np.array(out).reshape(-1, 1)

class DecisionNode(Node):
    """A Decision Node in a decision tree.
    
    Attributes:
        is_leaf (bool): Whether the node is a leaf or not.
        value (float): The value of the node.
        children (list): The children of the node.
        impurity (float): The impurity of the node.
        condition (tuple): The condition of the node.
    
    Methods:
        forwardpass(value): The forward pass of the node.
    """
    def __init__(self, depth, samples, impurity_function=gini) -> None:
        super().__init__(depth, samples, impurity_function, leaf_function=mode)

class RegressionNode(Node):
    """A Regression Node in a decision tree.
    
    Attributes:
        is_leaf (bool): Whether the node is a leaf or not.
        value (float): The value of the node.
        children (list): The children of the node.
        impurity (float): The impurity of the node.
        condition (tuple): The condition of the node.
    
    Methods:
        forwardpass(value): The forward pass of the node.
    """
    def __init__(self, depth, samples, impurity_function=np.var) -> None:
        super().__init__(depth, samples, impurity_function, leaf_function=np.mean)
