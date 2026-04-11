import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    a =[]
    for i in x:
        a.append(((np.e)**(i) - (np.e)**(-i))/((np.e)**(i) + (np.e)**(-i)))

    return a
    