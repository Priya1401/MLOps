import mlflow
from pathlib import Path

# Resolve absolute path to the directory ABOVE src/
BASE_DIR = Path(__file__).resolve().parent.parent

# Absolute path to mlruns folder
MLRUNS_DIR = BASE_DIR / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)

# Set MLflow tracking + registry to SAME absolute directory
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_registry_uri(f"file:{MLRUNS_DIR}")

mlflow.set_experiment("fastapi_mlops_lab")

print("MLflow tracking using:", MLRUNS_DIR)
