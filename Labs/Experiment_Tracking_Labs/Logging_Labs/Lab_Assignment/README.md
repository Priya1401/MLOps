
## FastAPI ML API — *Logging and Monitoring Update*

### Overview
This lab extends the **FastAPI Machine Learning Inference API** from previous assignments.
Originally, the lab exposed two scikit-learn models (Iris & Wine classifiers) as REST API endpoints.
In this update, the focus was on **implementing structured logging** to capture application behavior during training, prediction, and API requests.

---

### Features Implemented

#### **1. Model Training**

* `train.py` trains a **Decision Tree Classifier** on the *Iris dataset* and saves it as `iris_model.pkl`.
* `train_wine.py` trains a **Logistic Regression** model on the *Wine dataset* and saves it as `wine_model.pkl`.
* Both scripts now log:

  * Dataset loading and splitting
  * Model training progress
  * Model save confirmation
  * Any encountered errors

---

#### **2. Model Prediction**

* `predict.py` and `predict_wine.py` perform predictions using the trained models.
* Each now includes detailed logging for:

  * Model loading
  * Prediction execution
  * Prediction outputs
  * Errors during inference

---

#### **3. FastAPI Application**

* `main.py` hosts two active endpoints:

  * `/predict_iris` — accepts four numeric features to classify Iris flowers.
  * `/predict_wine` — accepts thirteen numeric features to classify Wine samples.
* Added a `/` health check and `/wine_metadata` endpoint for model introspection.
* Integrated structured logging for:

  * API startup and health checks
  * Incoming requests and prediction results
  * Exception handling for failed predictions

---

### **4. Logging System Integration (New for this Lab)**

#### File Added: `logger_config.py`

A centralized configuration file for the entire lab.

**Features:**

* Logs are written to both **console** and **file** (`../logs/app.log`)
* Uses **RotatingFileHandler** to prevent large log sizes
* Captures `INFO` level logs to console and `DEBUG` level logs to file
* Consistent log format:

  ```
  2025-11-03 15:42:10 — main — INFO — Received Wine input: [...]
  2025-11-03 15:42:10 — predict_wine — DEBUG — Predictions: [1]
  ```
  
---

### **How to Run**

#### Step 1: Setup environment

```bash
cd src
python3 -m venv .venv
source .venv/bin/activate      # (Mac/Linux)
pip install -r ../requirements.txt
```

#### Step 2: Train models

```bash
python train.py
python train_wine.py
```

#### Step 3: Run the API

```bash
uvicorn main:app --reload
```

#### Step 4: Open API Docs

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Example Endpoints

**Iris Prediction**

```bash
curl -X POST "http://127.0.0.1:8000/predict_iris" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

**Wine Prediction**

```bash
curl -X POST "http://127.0.0.1:8000/predict_wine" \
  -H "Content-Type: application/json" \
  -d '{"alcohol":13.2,"malic_acid":1.7,"ash":2.3,"alcalinity_of_ash":16.8,"magnesium":100,
       "total_phenols":2.2,"flavanoids":2.0,"nonflavanoid_phenols":0.3,"proanthocyanins":1.7,
       "color_intensity":5.0,"hue":1.0,"od280_od315_of_diluted_wines":3.0,"proline":1050}'
```

---

### **5. Output Verification**

**Logs available in `logs/app.log`:**

* API startup confirmation
* Dataset load and training steps
* User request details
* Model predictions and errors (if any)

Example:

```
2025-11-03 18:12:42 — main — INFO — Received Iris input: [[5.1, 3.5, 1.4, 0.2]]
2025-11-03 18:12:42 — predict — INFO — Loading iris_model.pkl for prediction
2025-11-03 18:12:42 — main — INFO — Iris prediction result: 0
```

---

### Outcome

This lab demonstrated:

* How to integrate **logging and monitoring** into an ML API workflow.
* The use of **centralized logging configuration** (`logger_config.py`) for consistency across scripts.
* The importance of **debug-level logs** for troubleshooting model loading, data validation, and runtime errors.
* How structured logging supports both **development debugging** and **operational monitoring**.
