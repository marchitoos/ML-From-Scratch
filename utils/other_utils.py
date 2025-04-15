import numpy as np

def mode(array, axis=0) -> np.ndarray:
    """Returns the mode of an array.
    
    Args:
        array (np.ndarray): Input array.
        axis (int): Axis along which to compute the mode. Default is 0.
    
    Returns:
        np.ndarray: The mode of the array.
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    if array.ndim == 1:
        unique_vals, counts = np.unique(array, return_counts=True)
        modes = unique_vals[counts == counts.max()]
        return np.mean(modes, axis=0)
    
    unique_vals, counts = np.unique(array, return_counts=True, axis=axis)
    print(unique_vals, counts)
    mode = np.mean(unique_vals[counts == np.max(counts, axis=axis)], axis=axis)
    return mode

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
