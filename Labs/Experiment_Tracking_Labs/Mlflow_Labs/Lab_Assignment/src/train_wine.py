import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .mlflow_config import *  # apply MLflow config


def train_wine():
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    with mlflow.start_run(run_name="wine_logreg"):
        pipe.fit(X_train, y_train)

        acc = pipe.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model", "LogisticRegression")

        signature = infer_signature(X_train, pipe.predict(X_train))

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="wine_model",
            signature=signature,
            registered_model_name="WineModel",
        )

        print("Wine model logged to MLflow.")


if __name__ == "__main__":
    train_wine()
