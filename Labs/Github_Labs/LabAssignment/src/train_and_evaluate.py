import os, json, hashlib
import pandas as pd
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"
THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.90"))  # deploy gate
DATA_SALT = os.getenv("DATA_SALT", "")                       # to simulate data change
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "false").lower() == "true"

def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def data_checksum(X, y, salt: str = "") -> str:
    # Hash the raw bytes + optional salt so you can simulate “new data”
    h = hashlib.md5()
    h.update(X.to_numpy().tobytes())
    h.update(y.to_numpy().tobytes())
    h.update(salt.encode("utf-8"))
    return h.hexdigest()

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test) -> float:
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def next_version() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)
    vfile = os.path.join(MODEL_DIR, "version.json")
    if os.path.exists(vfile):
        with open(vfile, "r") as f:
            data = json.load(f)
        version = int(data.get("version", 0)) + 1
    else:
        version = 1
    with open(vfile, "w") as f:
        json.dump({"version": version}, f)
    return version

def main():
    print(" Smart retraining + accuracy gate")
    X_train, X_test, y_train, y_test = load_data()

    # Checksum logic (no file dependency)
    checksum = data_checksum(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), DATA_SALT)
    checksum_file = os.path.join(MODEL_DIR, "checksum.txt")
    os.makedirs(MODEL_DIR, exist_ok=True)
    prev_checksum = open(checksum_file).read().strip() if os.path.exists(checksum_file) else None

    if not FORCE_RETRAIN and prev_checksum == checksum:
        print(" Data unchanged (by checksum). Skipping retraining.")
        print("HINT: Trigger retrain by setting FORCE_RETRAIN=true or changing DATA_SALT input.")
        return

    model = train_model(X_train, y_train)
    acc = evaluate(model, X_test, y_test)
    print(f" Accuracy: {acc:.4f}")

    version = next_version()
    model_path = os.path.join(MODEL_DIR, f"model_v{version}.joblib")
    dump(model, model_path)
    with open("metrics.json", "w") as f:
        json.dump({"version": version, "accuracy": acc}, f)

    with open(checksum_file, "w") as f:
        f.write(checksum)

    if acc < THRESHOLD:
        raise ValueError(f" Accuracy {acc:.4f} < threshold {THRESHOLD:.2f}. Blocking image publish.")
    print(f" Model v{version} passed gate and is ready for container build.")

if __name__ == "__main__":
    main()
