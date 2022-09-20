import argparse
from typing import List
import pandas as pd
import yaml

from src.utils.logs import get_logger

def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def filter_years(data: pd.DataFrame, offset: int) -> pd.DataFrame:
    max_year = data["year"].max()
    data = data[data["year"] == (max_year - offset)]
    return data

def filter_columns(
    data: pd.DataFrame, 
    features: List[str],
    target: List[str]
) -> pd.DataFrame:
    columns = [*target, *features]
    data = data[columns]
    return data

def filter_nas(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data

def clean_dataset(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = get_logger('MAKE_DATASET', log_level=config["base"]["log_level"])
    
    logger.info(f'Load and process dataset from {config["data_load"]["raw"]}')
    data = load_dataset(path=config["data_load"]["raw"])
    data = filter_years(data=data, offset=config["data_clean"]["offset"])
    data = filter_columns(
        data=data, 
        target=config["data_clean"]["target"],
        features=config["data_clean"]["features"]
    )
    data = filter_nas(data=data)
    
    logger.info(f'Save new dataset to {config["data_load"]["interim"]}')
    data.to_csv(config["data_load"]["interim"], index=False)

    return data

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    clean_dataset(config_path=args.config)