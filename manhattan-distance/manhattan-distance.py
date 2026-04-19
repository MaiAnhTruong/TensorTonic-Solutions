import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    distance = np.sum(np.abs(np.array(x) - np.array(y)))
    
    return float(distance)