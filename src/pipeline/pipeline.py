from typing import Dict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

def make_pipeline(config: Dict) -> Pipeline:

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
            ('lr', HistGradientBoostingRegressor(random_state=config["base"]["seed"]))
        ]
    )

    return pipeline