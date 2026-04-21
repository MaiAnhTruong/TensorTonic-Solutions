import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for _ in range(steps):
        # Linear output
        z = X @ w + b

        # Predicted probabilities
        y_pred = _sigmoid(z)

        # Errors
        error = y_pred - y

        # Gradients
        dw = (X.T @ error) / n_samples
        db = np.sum(error) / n_samples

        # Parameter update
        w -= lr * dw
        b -= lr * db

    return w, b