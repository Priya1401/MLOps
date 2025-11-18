from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel

from .predict import predict_data as predict_iris
from .predict_wine import predict_data as predict_wine, get_feature_meta as wine_meta
from .logger_config import logger

# Initialize FastAPI app
app = FastAPI(title="Iris & Wine Inference API")
logger.info("FastAPI application initialized successfully")

# ---------- Iris ----------
class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float


class IrisResponse(BaseModel):
    response: int


# ---------- Wine ----------
class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class WineResponse(BaseModel):
    response: int


# ---------- Endpoints ----------

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """
    Health check endpoint that also lists available models and their metadata.
    """
    logger.info("Health check endpoint called")
    try:
        wine_n, wine_feats = wine_meta()
        logger.debug(f"Wine model metadata retrieved: {wine_n} features")
        return {
            "status": "healthy",
            "models": {
                "iris": {"n_features": 4, "endpoint": "/predict_iris"},
                "wine": {
                    "n_features": wine_n,
                    "endpoint": "/predict_wine",
                    "feature_names": wine_feats,
                },
            },
        }
    except Exception as e:
        logger.exception("Error retrieving model metadata")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_iris", response_model=IrisResponse)
async def predict_iris_endpoint(iris_features: IrisData):
    """
    Predicts the Iris flower class based on input features.
    """
    try:
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width,
        ]]
        logger.info(f"Received Iris input: {features}")
        prediction = predict_iris(features)
        logger.info(f"Iris prediction result: {prediction[0]}")
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        logger.exception("Error occurred during Iris prediction")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wine_metadata")
def wine_metadata():
    """
    Returns Wine model metadata such as feature count and names.
    """
    try:
        logger.info("Fetching Wine model metadata")
        n, names = wine_meta()
        logger.debug(f"Wine features: {names}")
        return {"n_features": n, "feature_names": names}
    except Exception as e:
        logger.exception("Error retrieving Wine metadata")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_wine", response_model=WineResponse)
async def predict_wine_endpoint(b: WineData):
    """
    Predicts the Wine class based on chemical composition input features.
    """
    try:
        x = [[
            b.alcohol, b.malic_acid, b.ash, b.alcalinity_of_ash, b.magnesium,
            b.total_phenols, b.flavanoids, b.nonflavanoid_phenols, b.proanthocyanins,
            b.color_intensity, b.hue, b.od280_od315_of_diluted_wines, b.proline,
        ]]
        logger.info(f"Received Wine input: {x}")
        pred = predict_wine(x)
        logger.info(f"Wine prediction result: {pred[0]}")
        return WineResponse(response=int(pred[0]))
    except Exception as e:
        logger.exception("Error occurred during Wine prediction")
        raise HTTPException(status_code=500, detail=str(e))
