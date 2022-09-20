import yaml
import pandas as pd
import argparse

def create_features(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data = pd.read_csv(config["data_load"]["interim"])

    # Nothing for now

    data.to_csv(config["data_load"]["processed"], index=False)

    return data

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    create_features(config_path=args.config)