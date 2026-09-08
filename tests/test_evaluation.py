import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation import choose_threshold, evaluate_model, save_feature_explanation


def test_evaluate_model_returns_confusion_counts():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = DummyClassifier(strategy="uniform", random_state=42)
    model.fit(X, y)

    result = evaluate_model("dummy", model, X, y, threshold=0.5)

    assert {"precision", "recall", "f1", "roc_auc", "pr_auc"}.issubset(result)
    assert result["false_positives"] + result["true_negatives"] == 2
    assert result["false_negatives"] + result["true_positives"] == 2
    assert result["estimated_cost"] >= 0


def test_predictions_follow_threshold():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = DummyClassifier(strategy="prior")
    model.fit(X, y)

    low = evaluate_model("dummy", model, X, y, threshold=0.1)
    high = evaluate_model("dummy", model, X, y, threshold=0.9)

    assert low["predictions"].sum() >= high["predictions"].sum()


def test_choose_threshold_returns_a_candidate():
    X = np.arange(20).reshape(-1, 1)
    y = np.array([0] * 15 + [1] * 5)
    model = LogisticRegression(random_state=42).fit(X, y)

    threshold = choose_threshold(
        model,
        X,
        y,
        candidates=[0.2, 0.4, 0.6],
        false_positive_cost=5,
        false_negative_cost=150,
    )

    assert threshold in {0.2, 0.4, 0.6}


def test_logistic_explanation_preserves_direction(tmp_path):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=42).fit(X, y)

    save_feature_explanation(model, ["feature_a", "feature_b"], "logistic", tmp_path)
    explanation = (tmp_path / "logistic_feature_explanation.csv").read_text()

    assert "direction" in explanation
    assert "aumenta risco" in explanation or "reduz risco" in explanation
