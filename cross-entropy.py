### this file goes over how to implement cross-entropy from scratch ###

import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross-entropy loss.
    
    Parameters:
    y_true (numpy.ndarray): True labels (0 or 1).
    y_pred (numpy.ndarray): Predicted probabilities.

    Returns:
    float: Binary cross-entropy loss.
    """
    epsilon = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
loss = binary_cross_entropy(y_true, y_pred)
print(f'Binary Cross-Entropy Loss: {loss}')
