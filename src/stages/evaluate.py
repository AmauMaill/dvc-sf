
from typing import Dict
import yaml
import argparse
import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt

from src.report.metrics import make_metrics
from src.report.visualize import plot_actual_vs_predicted

def save_metrics(metrics: Dict, path: str):
    with open(path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

def save_plot(path: str):
    plt.savefig(path)

def evaluate_model(config_path: str) -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pipeline = joblib.load(config["train"]["model_path"])

    X_test = pd.read_csv(config["data_split"]["testset_x_path"])
    y_test = pd.read_csv(config["data_split"]["testset_y_path"])

    predicted = pipeline.predict(X_test)

    metrics = make_metrics(y_test.to_numpy(), predicted)
    save_metrics(metrics, config["evaluate"]["metrics_path"])

    plot_actual_vs_predicted(y_test.to_numpy(), predicted)
    save_plot(config["evaluate"]["figures_path"])

    return metrics



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)