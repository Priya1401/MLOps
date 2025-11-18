# FastAPI ML API with MLflow Tracking & Model Registry

## 1. Overview

This lab extends the previous **FastAPI Machine Learning Inference API** to integrate mlflow.

Originally, the project:
- Trained two scikit-learn models (Iris & Wine classifiers)
- Exposed them as REST API endpoints using FastAPI
- Added centralized logging via `logger_config.py`

In this lab, we **integrate MLflow** to:
- Track experiments, parameters, and metrics  
- Log and version models  
- Register models in the **MLflow Model Registry**  
- Load models from the registry in the FastAPI service (using the **Production** stage)

Logging is still enabled, but the main focus is now **experiment tracking and model lifecycle management**.

---

## 2. Project Structure

```text
Lab_Assignment/
├─ assets/
├─ logs/                   # application logs (app.log)
├─ mlruns/                 # MLflow tracking & model registry data
├─ model/                  # (unused now; kept from previous lab)
├─ src/
│   ├─ __init__.py
│   ├─ data.py             # iris dataset loading & splitting
│   ├─ logger_config.py    # centralized logging config
│   ├─ main.py             # FastAPI application
│   ├─ mlflow_config.py    # MLflow tracking/registry configuration
│   ├─ predict.py          # Iris model loading & prediction (from MLflow)
│   ├─ predict_wine.py     # Wine model loading & prediction (from MLflow)
│   ├─ train.py            # Train + register Iris model in MLflow
│   └─ train_wine.py       # Train + register Wine model in MLflow
├─ README.md
└─ requirements.txt
```

---

## 3. Key Components

### 3.1 MLflow Configuration — `src/mlflow_config.py`

- Configures **absolute** tracking and registry URIs to `Lab_Assignment/mlruns`.
- Ensures all scripts (training, FastAPI, and MLflow UI) share the same backend.
- Sets a common experiment:

```python
mlflow.set_experiment("fastapi_mlops_lab")
```

### 3.2 Model Training & Registration

#### `src/train.py` — Iris Model

- Loads the Iris dataset via `data.py`.
- Trains a **DecisionTreeClassifier**.
- Computes accuracy on a hold-out test set.
- Logs to MLflow:
  - Parameter: `max_depth`
  - Metric: `accuracy`
  - Model signature (input/output schema)
- Registers the model as:

```text
Registered Model Name: IrisModel
Artifact path: iris_model
```

#### `src/train_wine.py` — Wine Model

- Loads the scikit-learn Wine dataset.
- Builds a `Pipeline` of:
  - `StandardScaler`
  - `LogisticRegression(max_iter=1000)`
- Logs to MLflow:
  - Metric: `accuracy`
  - Param: `model="LogisticRegression"`
  - Model signature
- Registers the model as:

```text
Registered Model Name: WineModel
Artifact path: wine_model
```

Each training run creates a **new model version** in the MLflow Model Registry.

---

### 3.3 Prediction Modules

#### `src/predict.py` — Iris

- Loads the model from **MLflow Model Registry**:

```python
mlflow.pyfunc.load_model("models:/IrisModel/production")
```

- Takes a 2D array of shape `(n_samples, 4)` and returns predicted class labels.
- Logs:
  - Model loading event
  - Prediction results

#### `src/predict_wine.py` — Wine

- Loads the Wine model from:

```python
mlflow.pyfunc.load_model("models:/WineModel/production")
```

- Provides `get_feature_meta()` for metadata (feature count and names).
- Logs all predictions and model loading behavior.

---

### 3.4 FastAPI Application — `src/main.py`

Exposes the following endpoints:

- `GET /`
  - Health check
  - Returns available models and Wine feature metadata

- `POST /predict_iris`
  - Request body: sepal & petal measurements
  - Uses `predict.py` to call the **Production** Iris model from MLflow

- `GET /wine_metadata`
  - Returns Wine model feature count and feature names

- `POST /predict_wine`
  - Request body: 13 numeric chemical features
  - Uses `predict_wine.py` to call the **Production** Wine model

Logging is integrated for:
- API startup and health check
- Incoming request payloads
- Prediction outputs
- Exception handling

---

### 3.5 Logging System — `src/logger_config.py`

- Configures a root logger for the entire project.
- Handlers:
  - **Console** (INFO and above)
  - **Rotating file** at `logs/app.log` (DEBUG and above)
- Format:

```text
2025-11-03 15:42:10 — main — INFO — Received Wine input: [...]
2025-11-03 15:42:10 — predict_wine — DEBUG — Predictions: [1]
```

---

## 4. Environment Setup

> **Important:** This project assumes Python 3.10 in a virtual environment.  
> (MLflow and some ML libraries are not fully compatible with Python 3.12 yet.)

From the **MLOps_Github** root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate       # Windows (PowerShell/CMD)

cd Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab_Assignment
pip install -r requirements.txt
```

You should now see `(.venv)` in your terminal prompt.

---

## 5. Training & Registering Models with MLflow

All commands below are run from:

```bash
Lab_Assignment/
```

### 5.1 Train the Iris model

```bash
(.venv) Lab_Assignment % python -m src.train
```

Expected console messages include:

```text
Iris model logged to MLflow.
Created version '1' of model 'IrisModel'.
```

### 5.2 Train the Wine model

```bash
(.venv) Lab_Assignment % python -m src.train_wine
```

Expected console messages include:

```text
Wine model logged to MLflow.
Created version 'X' of model 'WineModel'.
```

Each execution creates a new model version in the registry.

---

## 6. Using the MLflow UI & Setting Models to Production

Start the MLflow UI:

```bash
(.venv) Lab_Assignment % mlflow ui --backend-store-uri file:mlruns --port 5001
```

Open: <http://127.0.0.1:5001>

1. Click the **Models** tab.
2. You should see:
   - `IrisModel`
   - `WineModel`
3. For each model:
   - Click on it
   - Select the latest version (e.g., `v1`, `v2`, `v3`, …)
   - In the *Stage* column, choose **Production**

After this, the URIs used in FastAPI:

```text
models:/IrisModel/production
models:/WineModel/production
```

will resolve correctly.

---

## 7. Running the FastAPI Service

From `Lab_Assignment` with the venv active:

```bash
(.venv) Lab_Assignment % uvicorn src.main:app --reload
```

The API will be available at:

- Base URL: <http://127.0.0.1:8000>
- Interactive docs (Swagger): <http://127.0.0.1:8000/docs>

---

## 8. Example Requests

### 8.1 Iris Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict_iris"   -H "Content-Type: application/json"   -d '{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      }'
```

Sample response:

```json
{
  "response": 0
}
```

### 8.2 Wine Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict_wine"   -H "Content-Type: application/json"   -d '{
        "alcohol": 13.2,
        "malic_acid": 1.7,
        "ash": 2.3,
        "alcalinity_of_ash": 16.8,
        "magnesium": 100,
        "total_phenols": 2.2,
        "flavanoids": 2.0,
        "nonflavanoid_phenols": 0.3,
        "proanthocyanins": 1.7,
        "color_intensity": 5.0,
        "hue": 1.0,
        "od280_od315_of_diluted_wines": 3.0,
        "proline": 1050
      }'
```

Sample response:

```json
{
  "response": 1
}
```

---

## 9. Logs and Observability

**Log file:** `logs/app.log`

Examples of log entries:

```text
2025-11-03 18:12:42 — main — INFO — Received Iris input: [[5.1, 3.5, 1.4, 0.2]]
2025-11-03 18:12:42 — predict — INFO — Loading Iris model from MLflow registry
2025-11-03 18:12:42 — main — INFO — Iris prediction result: 0
```

These logs record:
- API startup and health checks
- Dataset loading and training flow (from training scripts)
- Incoming request payloads
- Model prediction outputs
- Any exceptions during training or inference

---

## 10. Learning Outcomes

This lab demonstrates how to:

- Extend a basic FastAPI ML API into an **MLOps-ready service**.
- Use **MLflow Tracking** to log parameters, metrics, and models.
- Use the **MLflow Model Registry** to manage multiple model versions and stages.
- Load models into an API **directly from the registry** using stage-based URIs.
- Maintain **centralized logging** for easier debugging and observability in ML systems.
