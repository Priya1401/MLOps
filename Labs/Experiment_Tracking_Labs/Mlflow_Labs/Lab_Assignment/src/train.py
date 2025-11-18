import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from .data import load_data, split_data
from .mlflow_config import *  # apply MLflow config (tracking URI, experiment)


def fit_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    with mlflow.start_run(run_name="iris_decision_tree"):
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=3, random_state=12)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("accuracy", acc)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            artifact_path="iris_model",
            signature=signature,
            registered_model_name="IrisModel",
        )

        print("Iris model logged to MLflow.")


if __name__ == "__main__":
    fit_model()
