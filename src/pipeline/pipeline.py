from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

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