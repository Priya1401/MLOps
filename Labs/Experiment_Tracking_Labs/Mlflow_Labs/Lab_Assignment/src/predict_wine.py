import numpy as np
from sklearn.datasets import load_wine
import mlflow
from .logger_config import logger

# Cache items so the model isn't reloaded every request
_model = None
_n_features = None
_feature_names = None


def get_feature_meta():
    """
    Return (n_features, feature_names) for Wine dataset.
    Used by root '/' and /wine_metadata endpoints.
    """
    global _n_features, _feature_names

    if _n_features is not None and _feature_names is not None:
        return _n_features, _feature_names

    logger.info("Loading wine feature metadata from sklearn.datasets.load_wine()")
    wine = load_wine()
    _n_features = wine.data.shape[1]
    _feature_names = list(wine.feature_names)
    return _n_features, _feature_names


def _load_model():
    """
    Load Wine model from MLflow Model Registry (Production Stage).
    """
    logger.info("Loading Wine model from MLflow registry: models:/WineModel/production")
    model = mlflow.pyfunc.load_model("models:/WineModel/production")
    logger.info("Wine model loaded successfully")
    return model


def predict_data(X):
    """
    Predict wine class.
    """
    global _model
    if _model is None:
        _model = _load_model()

    logger.info("Running Wine model prediction")
    X = np.asarray(X, dtype=float)
    preds = _model.predict(X)
    logger.debug(f"Wine predictions: {preds}")
    return preds
