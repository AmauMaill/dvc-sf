from typing import Dict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklego.linear_model import LADRegression
from sklearn.pipeline import Pipeline

def make_pipeline() -> Pipeline:
    categorical_transformer = Pipeline(
    [
        ('ohe', OneHotEncoder(sparse=False))
    ]
    )

    preprocessor = ColumnTransformer([
        ("categorical", categorical_transformer, ["county"])
    ])

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ('lr', LADRegression())
        ]
    )

    return pipeline