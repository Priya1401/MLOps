# DVC Integration Lab – FastAPI Iris & Wine Model Project

## Overview

Integrated **Data Version Control (DVC)** into an existing FastAPI machine learning project that serves predictions for two models:

- **Iris Classification Model** (`iris_model.pkl`)
- **Wine Classification Model** (`wine_model.pkl`)

The goal of the lab was to demonstrate:

- Model versioning  
- Artifact tracking  
- Use of DVC remotes  
- Reproducible ML workflows  

Since this FastAPI project does not rely on downloadable datasets (it uses sklearn built‑in datasets), the most meaningful use of DVC is to **version-control the trained model artifacts**.

---

## Project Structure

```
Lab_Assignment/
│── model/
│     ├── iris_model.pkl
│     └── wine_model.pkl
│
│── src/
│     ├── data.py
│     ├── train.py
│     ├── train_wine.py
│     ├── predict.py
│     ├── predict_wine.py
│     └── main.py
│
│── dvc_store/
│── .dvc/
│── model/*.dvc
│── README.md
```

---

## Step 1 — Initialize Git and DVC

```
git init
git add .
git commit -m "Initial commit"
dvc init
```

---

## Step 2 — Track ML Model Artifacts with DVC

```
dvc add model/iris_model.pkl
dvc add model/wine_model.pkl
git add model/*.dvc model/.gitignore
git commit -m "Track Iris and Wine models with DVC"
```

---

## Step 3 — Configure a Local DVC Remote (No GCP Needed)

```
dvc remote add -d localstore ./dvc_store
```

---

## Step 4 — Push Artifacts to Remote

```
dvc push
```

This stores the model artifacts inside `dvc_store/`, while Git tracks only the `.dvc` pointer files.

---

## Step 5 — Reproducing Model Versions

To restore a previous version:

```
git checkout <commit_hash>
dvc pull
```

---

## FastAPI With Versioned Models

FastAPI loads the models normally:

```python
model = joblib.load("../model/iris_model.pkl")
```

Because DVC manages the file's actual version, each Git commit corresponds to a specific model version.

---

## Summary

With this setup:

- DVC is initialized  
- Model artifacts are versioned  
- A local DVC remote is used  
- Push/pull workflow is enabled  
- FastAPI serves DVC-tracked models  
- The project is now reproducible and aligned with MLOps best practices  

This completes the required DVC integration for the lab.

