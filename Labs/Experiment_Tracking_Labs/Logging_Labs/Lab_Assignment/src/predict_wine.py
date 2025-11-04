import joblib
import numpy as np
from logger_config import logger

logger.info("Loading wine_model.pkl once at import")
_bundle = joblib.load("../model/wine_model.pkl")
_model = _bundle["model"]
_feature_names = _bundle.get("feature_names", None)

def get_feature_meta():
    logger.debug("Fetching feature metadata for Wine model")
    if _feature_names is not None:
        return len(_feature_names), _feature_names
    return _model.n_features_in_, None

def predict_data(X):
    logger.info("Running Wine model prediction")
    X = np.asarray(X, dtype=float)
    preds = _model.predict(X)
    logger.debug(f"Predictions: {preds}")
    return preds
