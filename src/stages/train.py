
import yaml
import argparse
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from src.utils.logs import get_logger

def make_pipeline() -> Pipeline:
    categorical_transformer = Pipeline(
    [
        ('ohe', OneHotEncoder())
    ]
    )

    preprocessor = ColumnTransformer([
        ("categorical", categorical_transformer, ["county"])
    ])

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ('lr', LinearRegression())
        ]
    )

    return pipeline

def train_model(config_path: str) -> Pipeline:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = get_logger('SPLIT_DATA', log_level=config["base"]["log_level"])

    logger.info('Create pipeline / model')
    pipeline = make_pipeline()

    logger.info('Load train datasets')
    X_train = pd.read_csv(config["data_split"]["trainset_x_path"])
    y_train = pd.read_csv(config["data_split"]["trainset_y_path"])

    logger.info('Fit the pipeline / model')
    pipeline.fit(X_train, y_train)

    logger.info(f'Save the pipeline / model to {config["train"]["model_path"]}')
    joblib.dump(pipeline, config["train"]["model_path"])

    return pipeline



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)