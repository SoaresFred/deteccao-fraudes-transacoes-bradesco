import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .evaluation import (
        choose_threshold,
        evaluate_model,
        save_curves,
        save_evaluation_plots,
        save_feature_explanation,
    )
    from .models import build_models
except ImportError:  # permite python src/main.py
    from evaluation import choose_threshold, evaluate_model, save_curves, save_evaluation_plots, save_feature_explanation
    from models import build_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def run(data_path: str, output_dir: str = "outputs"):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    required = {"Time", "Amount", "Class"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("A coluna Class deve conter apenas 0 e 1.")

    df = df.sort_values("Time").reset_index(drop=True)

    LOGGER.info("Distribuição das classes:\n%s", df["Class"].value_counts(normalize=True))
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Class")
    plt.title("Distribuição das classes")
    plt.xlabel("Classe (0 = normal, 1 = fraude)")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    # O dataset não tem data real; extraímos uma hora aproximada e a codificamos ciclicamente.
    hour = (df["Time"] % 86400) / 3600
    df["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    # Divisão out-of-time: passado para treino, período intermediário para validação
    # e período posterior para teste. Assim, o futuro nunca influencia o passado.
    split_train = int(len(df) * 0.60)
    split_validation = int(len(df) * 0.80)
    X_train, X_validation, X_test = (
        X.iloc[:split_train],
        X.iloc[split_train:split_validation],
        X.iloc[split_validation:],
    )
    y_train, y_validation, y_test = (
        y.iloc[:split_train],
        y.iloc[split_train:split_validation],
        y.iloc[split_validation:],
    )
    if y_train.nunique() < 2 or y_validation.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "A divisão temporal precisa conter as classes 0 e 1 em treino, validação e teste."
        )

    results = []
    for name, model in build_models(random_state=42).items():
        LOGGER.info("Treinando modelo: %s", name)
        model.fit(X_train, y_train)
        artifact_name = name.lower().replace(" ", "_").replace("ã", "a")
        joblib.dump(model, output_dir / f"{artifact_name}.pkl")
        save_feature_explanation(model, X_train.columns, name, output_dir)
        selected_threshold = choose_threshold(
            model,
            X_validation,
            y_validation,
            false_positive_cost=5.0,
            false_negative_cost=150.0,
        )
        LOGGER.info("Threshold selecionado na validação para %s: %.2f", name, selected_threshold)

        for threshold in (0.50, selected_threshold):
            label = name if threshold == 0.50 else f"{name} (threshold validado)"
            result = evaluate_model(
                label,
                model,
                X_test,
                y_test,
                threshold=threshold,
                false_positive_cost=5.0,
                false_negative_cost=150.0,
            )
            save_evaluation_plots(result, y_test, output_dir)
            results.append(result)

    save_curves([r for r in results if r["threshold"] == 0.50], y_test, output_dir)
    export_data = [
        {k: v for k, v in r.items() if k not in {"predictions", "probabilities"}}
        for r in results
    ]
    pd.DataFrame(export_data).to_csv(output_dir / "model_metrics.csv", index=False)
    LOGGER.info("Execução concluída. Métricas e gráficos salvos em %s.", output_dir)
    return pd.DataFrame(export_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecção de fraudes em transações")
    parser.add_argument("--data", required=True, help="Caminho para creditcard.csv")
    parser.add_argument("--output", default="outputs", help="Diretório de saída")
    args = parser.parse_args()
    run(args.data, args.output)
