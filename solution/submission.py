# solution/submission.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class Model:
    """
    Starter submission.

    The ingestion program calls:
      - fit(X_train, y_train)
      - predict(X_test)

    y contains strings: Low / Medium / High / Critical
    """

    def __init__(self):
        self.pipeline = None

    def fit(self, X_train, y_train):
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
        num_cols = [c for c in X_train.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        # =========================
        # YOUR CODE HERE
        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=42
        )
        # =========================

        self.pipeline = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", model),
        ])

        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        return self.pipeline.predict(X_test)


def get_model():
    return Model()