import yaml
import pandas as pd
import argparse

from src.utils.logs import get_logger

def create_features(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = get_logger('BUILD_FEATURES', log_level=config["base"]["log_level"])

    logger.info(f'Load and process dataset from {config["data_load"]["interim"]}')
    data = pd.read_csv(config["data_load"]["interim"])

    # Nothing for now
    logger.info("No steps executed for the moment")

    logger.info(f'Save new dataset to {config["data_load"]["processed"]}')
    data.to_csv(config["data_load"]["processed"], index=False)

    return data

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    create_features(config_path=args.config)