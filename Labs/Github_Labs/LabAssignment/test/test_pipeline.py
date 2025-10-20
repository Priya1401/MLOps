from src.train_and_evaluate import load_data, data_checksum, train_model, evaluate
import pandas as pd
from hashlib import md5

def test_load_data_shapes():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) + len(X_test) > 0
    assert len(y_train) + len(y_test) > 0

def test_checksum_changes_with_salt():
    X_train, X_test, y_train, y_test = load_data()
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    c1 = data_checksum(X, y, "")
    c2 = data_checksum(X, y, "salt")
    assert c1 != c2

def test_training_and_eval_runs():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    acc = evaluate(model, X_test, y_test)
    assert 0.0 <= acc <= 1.0
