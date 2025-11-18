import mlflow
from .logger_config import logger


def predict_data(X):
    logger.info("Loading Iris model from MLflow registry")
    model = mlflow.pyfunc.load_model("models:/IrisModel/production")
    preds = model.predict(X)
    logger.debug(f"Iris prediction output: {preds}")
    return preds
