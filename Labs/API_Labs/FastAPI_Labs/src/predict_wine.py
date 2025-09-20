import joblib
import numpy as np

_bundle = joblib.load("../model/wine_model.pkl")  # load once at import
_model = _bundle["model"]
_feature_names = _bundle.get("feature_names", None)

def get_feature_meta():
    if _feature_names is not None:
        return len(_feature_names), _feature_names
    return _model.n_features_in_, None

def predict_data(X):
    """
    Predict Wine class for input vectors.
    Args:
        X (array-like, shape [n_samples, 13])
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    X = np.asarray(X, dtype=float)
    return _model.predict(X)
