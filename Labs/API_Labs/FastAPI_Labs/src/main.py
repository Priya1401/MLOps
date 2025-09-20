from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data as predict_iris
from predict_wine import predict_data as predict_wine, get_feature_meta as wine_meta

app = FastAPI(title="Iris & Wine Inference API")

# ---------- Iris ----------
class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response: int

# ---------- Wine ----------
# Using explicit fields for clarity; order matters in the list we build
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

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    wine_n, wine_feats = wine_meta()
    return {
        "status": "healthy",
        "models": {
            "iris": {"n_features": 4, "endpoint": "/predict_iris"},
            "wine": {"n_features": wine_n, "endpoint": "/predict_wine", "feature_names": wine_feats},
        },
    }

@app.post("/predict_iris", response_model=IrisResponse)
async def predict_iris_endpoint(iris_features: IrisData):
    try:
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]]
        prediction = predict_iris(features)
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wine_metadata")
def wine_metadata():
    n, names = wine_meta()
    return {"n_features": n, "feature_names": names}

@app.post("/predict_wine", response_model=WineResponse)
async def predict_wine_endpoint(b: WineData):
    try:
        x = [[
            b.alcohol, b.malic_acid, b.ash, b.alcalinity_of_ash, b.magnesium,
            b.total_phenols, b.flavanoids, b.nonflavanoid_phenols, b.proanthocyanins,
            b.color_intensity, b.hue, b.od280_od315_of_diluted_wines, b.proline
        ]]
        pred = predict_wine(x)
        return WineResponse(response=int(pred[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
