import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """

    if len(y_pred) != len(y_true):
        return None
    n = len(y_pred)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return (1/n) * sum((y_pred - y_true)**2)
