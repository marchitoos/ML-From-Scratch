import numpy as np

def mode(array) -> np.ndarray:
    """Returns the mode of an array.
    
    Args:
        array (np.ndarray): Input array.
    
    Returns:
        np.ndarray: The mode of the array.
    """
    values, counts = np.unique(array, return_counts=True)
    modes = values[counts == counts.max()]
    print(len(modes))
    print(modes)
    return np.mean(modes)

def gini(array) -> float:
    """Calculate the Gini impurity of an array.
    
    Args:
        array (np.ndarray): Input array.
    
    Returns:
        float: The Gini impurity of the array.
    """
    _, counts = np.unique(array, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)
