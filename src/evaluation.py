import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.lower()).strip("_")


def evaluate_model(name: str, model, X_eval, y_eval, threshold: float = 0.5) -> dict:
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(
        y_eval, predictions, output_dict=True, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_eval, predictions, labels=[0, 1]).ravel()
    has_both_classes = len(set(y_eval)) > 1

    return {
        "model": name,
        "threshold": threshold,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "roc_auc": roc_auc_score(y_eval, probabilities) if has_both_classes else 0.5,
        "pr_auc": average_precision_score(y_eval, probabilities) if has_both_classes else 0.0,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def choose_threshold(model, X_validation, y_validation, candidates=None, min_precision=0.50):
    candidates = candidates or [round(i / 100, 2) for i in range(10, 50, 5)]
    validation_results = [
        evaluate_model("validation", model, X_validation, y_validation, threshold=t)
        for t in candidates
    ]
    eligible = [r for r in validation_results if r["precision"] >= min_precision]
    pool = eligible or validation_results
    best = max(pool, key=lambda r: (r["recall"], r["f1"], r["precision"]))
    return float(best["threshold"])


def save_evaluation_plots(result: dict, y_eval, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(
        f"{result['model']}_threshold_{int(result['threshold'] * 100)}"
    )
    matrix = confusion_matrix(y_eval, result["predictions"], labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de confusão — {result['model']} (t={result['threshold']:.2f})")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_dir / f"{clean_name}_confusion_matrix.png", dpi=150)
    plt.close()


def save_feature_explanation(model, feature_names, model_name: str, output_dir: Path):
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
        if estimator is not None and hasattr(estimator, "coef_"):
            values = abs(estimator.coef_[0])

    if values is None:
        return

    importance = pd.Series(values, index=feature_names, name="importance")
    importance = importance.sort_values(ascending=False)
    importance.to_csv(output_dir / f"{sanitize_filename(model_name)}_feature_importance.csv")

    top = importance.head(15).sort_values()
    plt.figure(figsize=(8, 6))
    top.plot(kind="barh", color="#2563eb")
    plt.title(f"Variáveis mais influentes — {model_name}")
    plt.xlabel("Importância absoluta")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{sanitize_filename(model_name)}_feature_importance.png",
        dpi=150,
    )
    plt.close()


def save_curves(results: list, y_eval, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plotted = False
    for result in results:
        if len(set(y_eval)) > 1:
            fpr, tpr, _ = roc_curve(y_eval, result["probabilities"])
            plt.plot(fpr, tpr, label=f"{result['model']} (AUC={result['roc_auc']:.3f})")
            plotted = True
    if plotted:
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Aleatório")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Recall / taxa de verdadeiros positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for result in results:
        precision, recall, _ = precision_recall_curve(
            y_eval, result["probabilities"]
        )
        plt.plot(recall, precision, label=f"{result['model']} (AP={result['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()
