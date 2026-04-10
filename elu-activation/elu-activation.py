import math

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    result = []
    for value in x:
        if value > 0:
            result.append(value)
        else:
            result.append(alpha * (math.exp(value) - 1))
    return result