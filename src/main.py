import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

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

    LOGGER.info("Distribuição das classes:\n%s", df["Class"].value_counts(normalize=True))
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Class")
    plt.title("Distribuição das classes")
    plt.xlabel("Classe (0 = normal, 1 = fraude)")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    # O dataset não tem data real; extraímos apenas a hora cíclica aproximada.
    df["Hour"] = (df["Time"] % 86400) / 3600
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    # 60% treino, 20% validação para escolher threshold, 20% teste final intocado.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    results = []
    for name, model in build_models(random_state=42).items():
        LOGGER.info("Treinando modelo: %s", name)
        model.fit(X_train, y_train)
        save_feature_explanation(model, X_train.columns, name, output_dir)
        selected_threshold = choose_threshold(
            model, X_validation, y_validation, min_precision=0.50
        )
        LOGGER.info("Threshold selecionado na validação para %s: %.2f", name, selected_threshold)

        for threshold in (0.50, selected_threshold):
            label = name if threshold == 0.50 else f"{name} (threshold validado)"
            result = evaluate_model(label, model, X_test, y_test, threshold=threshold)
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
