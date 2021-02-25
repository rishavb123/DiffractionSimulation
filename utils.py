"""A file to hold any utility functions required"""
import numpy as np

def one_hot(n, l=0) -> np.array:
    """One hot encodes a scalar

    Args:
        n (int): The number to one hot encode
        l (int): The length of the one hot encoded numpy array. Defaults to n + 1.

    Returns:
        np.array: The one hot encoded array of length l
    """    
    if l < n + 1:
        l = n + 1
    arr = np.zeros(l)
    arr[n] = 1
    return arr

def almost_equals(a, b, eps=1e-5) -> bool:
    """Checks if two values are close up to a difference of eps

    Args:
        a (float): The first value
        b (float): The second value
        eps (float, optional): The maximum allowed difference between a and b. Defaults to 1e-5.

    Returns:
        bool: Whether or not the two values have a difference greater than epsilon
    """    
    return abs(a - b) < eps