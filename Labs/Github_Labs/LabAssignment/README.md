# Lab Assignment – Smart Retraining and Docker CI/CD with GitHub Actions

## Overview
This lab builds on previous projects by adding an automated and intelligent CI/CD pipeline entirely within GitHub.  
It trains a Random Forest model using the Iris dataset from scikit-learn and introduces:
- **Smart retraining** — the workflow detects if data has changed and retrains only when needed.
- **Accuracy gate** — deployment stops automatically if accuracy falls below a threshold.
- **Docker automation** — the trained model is packaged and published to GitHub Container Registry.

## How It Works
1. GitHub Actions runs on each push or manual trigger.
2. Unit tests validate the code.
3. The model is trained and evaluated.
4. If accuracy ≥ 0.90, a Docker image is built and pushed to GHCR.
5. Artifacts (model and metrics) are stored for version tracking.

## How to Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run training and evaluation
python src/train_and_evaluate.py

# Run tests
pytest
```

## How to Run on GitHub
- Push any changes to the `main` branch **or**
- Manually trigger the workflow from the **Actions** tab  
  (optional inputs: `force_retrain`, `preview_only`, `data_salt`)

## Output
- Trained model files saved under `models/`
- Accuracy metrics in `metrics.json`
- Docker image available in **GitHub → Packages → Container Registry**
