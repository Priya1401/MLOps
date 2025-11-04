import joblib
from logger_config import logger

def predict_data(X):
    logger.info("Loading iris_model.pkl for prediction")
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    logger.debug(f"Prediction completed: {y_pred}")
    return y_pred
