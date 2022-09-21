import boto3
import pandas as pd
import argparse
import yaml

from src.utils.logs import get_logger

def make_dataset(config_path: str = None) -> pd.DataFrame:

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = get_logger('MAKE_DATASET', log_level=config["base"]["log_level"])

    logger.info('Create s3 client')
    client = boto3.client(
        "s3"
    )

    logger.info('Get dataset from s3')
    response = client.get_object(Bucket="dvc-sf", Key="rent.csv")
    data = pd.read_csv(response.get("Body"), sep=",")

    logger.info(f'Save dataset to {config["data_load"]["raw"]}')
    data.to_csv(config["data_load"]["raw"], index=False)

    return data

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    make_dataset(config_path=args.config)