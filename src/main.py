import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from .evaluation import evaluate_model, save_curves, save_evaluation_plots
from .models import build_models


def run(data_path: str, output_dir: str = "outputs"):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    required = {"Time", "Amount", "Class"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    print("Distribuição das classes:")
    print(df["Class"].value_counts(normalize=True).rename("proporção"))

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Class")
    plt.title("Distribuição das classes")
    plt.xlabel("Classe (0 = normal, 1 = fraude)")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    X = df.drop(columns="Class")
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    results = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test, output_dir)
        result_low_threshold = evaluate_model(
            f"{name} (limiar 0.30)", model, X_test, y_test, output_dir, threshold=0.30
        )
        print(f"\n{name} — limiar 0.50")
        print({k: round(result[k], 4) for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")})
        print(f"{name} — limiar 0.30")
        print({k: round(result_low_threshold[k], 4) for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")})
        save_evaluation_plots(result, y_test, output_dir)
        results.append(result)

    save_curves(results, y_test, output_dir)
    pd.DataFrame(
        [{k: v for k, v in result.items() if k not in {"predictions", "probabilities"}} for result in results]
    ).to_csv(output_dir / "model_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecção de fraudes em transações")
    parser.add_argument("--data", required=True, help="Caminho para creditcard.csv")
    parser.add_argument("--output", default="outputs", help="Diretório de saída")
    args = parser.parse_args()
    run(args.data, args.output)
