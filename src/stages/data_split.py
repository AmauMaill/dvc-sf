import yaml
import argparse

import pandas as pd

from sklearn.model_selection import train_test_split

def save_split(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path, index=False)

def make_data_split(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data = pd.read_csv(config["data_load"]["processed"])
    features = config["data_clean"]["features"]
    target = config["data_clean"]["target"]

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=config["base"]["seed"],
        test_size=config["data_split"]["test_size"]
    )

    save_split(X_train, config["data_split"]["trainset_x_path"])
    save_split(y_train, config["data_split"]["trainset_y_path"])
    save_split(X_test, config["data_split"]["testset_x_path"])
    save_split(y_test, config["data_split"]["testset_y_path"])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    make_data_split(config_path=args.config)