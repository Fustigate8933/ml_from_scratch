import numpy as np

def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray):
    """
    Compute Mean Squared Error between predictions and true values.
    """
    return np.mean((y_pred - y_true) ** 2)

