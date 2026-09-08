from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(name, model, X_test, y_test, output_dir: Path, threshold=0.5):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return {
        "model": name,
        "threshold": threshold,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def save_evaluation_plots(result, y_test, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    name = result["model"].lower().replace(" ", "_")
    matrix = confusion_matrix(y_test, result["predictions"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de confusão — {result['model']}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()


def save_curves(results, y_test, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result["probabilities"])
        plt.plot(fpr, tpr, label=f"{result['model']} (AUC={result['roc_auc']:.3f})")
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
        precision, recall, _ = precision_recall_curve(y_test, result["probabilities"])
        plt.plot(recall, precision, label=f"{result['model']} (AP={result['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()
