from typing import Dict

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

def make_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    metrics = {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mean_squared_error(actual, predicted),
    }

    return metrics