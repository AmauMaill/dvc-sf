import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_data_actual_vs_predicted(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> pd.DataFrame:
    """
    Given the actual values and the predicted values, return a dataframe.

    Args:
        actual (np.ndarray): The actual values (ex: y_test)
        predicted (np.ndarray): The predicted values
    """
    data = {
        "actual": actual.ravel(),
        "predicted": predicted.ravel()
    }

    data = pd.DataFrame.from_dict(data)

    return data

def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray):
    """
    Given the actual values and the predicted values, plot a scatter plot.

    Args:
        actual (np.ndarray): The actual values (ex: y_test)
        predicted (np.ndarray): The predicted values
    """
    plt.scatter(actual, predicted, c='crimson')

    p1 = max(max(predicted), max(actual))
    p2 = min(min(predicted), min(actual))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')