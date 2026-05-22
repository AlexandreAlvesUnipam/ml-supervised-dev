import argparse
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import save_model


def get_paths():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_path = repo_root / "data" / "flores_iris.csv"
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return data_path, artifacts_dir


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "Id" not in df.columns or "Species" not in df.columns:
        raise ValueError("O CSV deve conter as colunas 'Id' e 'Species'.")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def prepare_data(df: pd.DataFrame):
    df = df.drop(columns=["Id"])
    X = df.drop(columns=["Species"])
    y = df["Species"].map(
        {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2,
        }
    )
    if y.isnull().any():
        raise ValueError("Existem valores de espécie desconhecidos no dataset.")
    return X, y


def train_and_evaluate(X_train, X_val, y_train, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)
    report = classification_report(y_val, y_pred, target_names=["setosa", "versicolor", "virginica"])
    accuracy = accuracy_score(y_val, y_pred)
    confusion = confusion_matrix(y_val, y_pred)

    return model, scaler, accuracy, report, confusion


def run(data_path: Path | None = None):
    default_data_path, artifacts_dir = get_paths()
    data_path = default_data_path if data_path is None else data_path
    df = load_data(data_path)

    holdout = df.sample(n=10, random_state=42)
    remainder = df.drop(index=holdout.index).reset_index(drop=True)

    X_remainder, y_remainder = prepare_data(remainder)
    X_holdout, y_holdout = prepare_data(holdout)

    X_train, X_val, y_train, y_val = train_test_split(
        X_remainder,
        y_remainder,
        test_size=0.2,
        random_state=42,
        stratify=y_remainder,
    )

    model, scaler, accuracy, report, confusion = train_and_evaluate(X_train, X_val, y_train, y_val)

    save_model(model, artifacts_dir / "iris_logistic_multiclass.pkl")
    save_model(scaler, artifacts_dir / "iris_scaler.pkl")

    print("=== Avaliação do modelo multiclass ===")
    print(f"Acurácia de validação: {accuracy:.4f}\n")
    print(report)
    print("Matriz de confusão:\n", confusion)

    X_holdout_scaled = scaler.transform(X_holdout)
    holdout_pred = model.predict(X_holdout_scaled)

    print("\n=== Predições sobre amostra holdout (10 exemplos) ===")
    for idx, (true_value, pred_value) in enumerate(zip(y_holdout, holdout_pred), 1):
        print(f"{idx:02d}. verdadeiro={true_value}  previsto={pred_value}")

    print(
        "\nObservação: este modelo resolve a atividade adequadamente porque treina uma classificação multiclasses em vez de tratar apenas Iris-setosa vs outras espécies."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo multiclasse sobre o dataset Iris.")
    parser.add_argument(
        "--data",
        default=None,
        help="Caminho para o arquivo CSV do dataset (padrão: data/flores_iris.csv).",
    )
    args = parser.parse_args()
    run(Path(args.data).resolve() if args.data else None)
