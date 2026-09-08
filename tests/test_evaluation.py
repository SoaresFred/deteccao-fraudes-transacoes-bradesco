import numpy as np
from sklearn.dummy import DummyClassifier

from src.evaluation import evaluate_model


def test_evaluate_model_returns_confusion_counts():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = DummyClassifier(strategy="uniform", random_state=42)
    model.fit(X, y)

    result = evaluate_model("dummy", model, X, y, threshold=0.5)

    assert {"precision", "recall", "f1", "roc_auc", "pr_auc"}.issubset(result)
    assert result["false_positives"] + result["true_negatives"] == 2
    assert result["false_negatives"] + result["true_positives"] == 2


def test_predictions_follow_threshold():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = DummyClassifier(strategy="prior")
    model.fit(X, y)

    low = evaluate_model("dummy", model, X, y, threshold=0.1)
    high = evaluate_model("dummy", model, X, y, threshold=0.9)

    assert low["predictions"].sum() >= high["predictions"].sum()
