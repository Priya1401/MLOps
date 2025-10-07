import os
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    """Loads the Iris dataset and serializes it."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    serialized_data = pickle.dumps(df)
    return serialized_data


def preprocess_data(data):
    """Splits and scales the dataset."""
    df = pickle.loads(data)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    serialized = pickle.dumps((X_train_scaled, X_test_scaled, y_train, y_test))
    return serialized


def train_model(data):
    """Trains a Logistic Regression model and saves it."""
    X_train, X_test, y_train, y_test = pickle.loads(data)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    output_dir = os.path.join(os.path.dirname(__file__), "../model")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "iris_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return pickle.dumps((model_path, X_test, y_test))


def test_model(data):
    """Loads the model and evaluates accuracy."""
    model_path, X_test, y_test = pickle.loads(data)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model accuracy on test set: {acc:.3f}")
    return acc
