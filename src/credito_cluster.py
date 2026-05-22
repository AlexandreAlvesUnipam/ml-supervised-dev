import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

STUDENT_NAME = "Nome Completo"


def get_paths():
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "credito.csv"
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return data_path, artifacts_dir


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    df = pd.read_csv(data_path)
    return df


def build_preprocessor(df: pd.DataFrame) -> Pipeline:
    numeric_features = [
        "Gender",
        "Age",
        "Debt",
        "Married",
        "BankCustomer",
        "YearsEmployed",
        "PriorDefault",
        "Employed",
        "CreditScore",
        "DriversLicense",
        "ZipCode",
        "Income",
    ]
    categorical_features = ["Industry", "Ethnicity", "Citizen"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def prepare_features(df: pd.DataFrame, preprocessor: Pipeline):
    features = df.drop(columns=["Approved"]).copy()
    X = preprocessor.fit_transform(features)
    return X


def find_best_k(X: np.ndarray, k_min: int = 2, k_max: int = 10):
    inertia = []
    silhouette_scores = []
    ks = list(range(k_min, k_max + 1))

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        inertia.append(model.inertia_)
        if len(np.unique(labels)) > 1:
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(float("nan"))

    best_k = ks[int(np.nanargmax(silhouette_scores))]
    return ks, inertia, silhouette_scores, best_k


def fit_kmeans(X: np.ndarray, n_clusters: int):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    return model, labels


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("Cluster").agg(
        count=("Cluster", "size"),
        income_mean=("Income", "mean"),
        age_mean=("Age", "mean"),
        debt_mean=("Debt", "mean"),
        credit_score_mean=("CreditScore", "mean"),
        years_employed_mean=("YearsEmployed", "mean"),
        zip_mean=("ZipCode", "mean"),
        approved_rate=("Approved", "mean"),
    )
    summary = summary.round(2)
    return summary


def save_reports(df: pd.DataFrame, summary: pd.DataFrame, artifacts_dir: Path):
    df.to_csv(artifacts_dir / "credito_cluster_assignments.csv", index=False)
    summary.to_csv(artifacts_dir / "credito_cluster_summary.csv")


def create_pdf_report(
    artifacts_dir: Path,
    ks,
    inertia,
    silhouette_scores,
    best_k,
    summary,
    df,
    labels,
    X,
):
    pdf_path = artifacts_dir / "credito_cluster_analysis.pdf"
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ks, inertia, marker="o", linewidth=2)
        ax.set_title("Método do Cotovelo: Inércia vs Número de Grupos")
        ax.set_xlabel("Número de grupos (k)")
        ax.set_ylabel("Inércia")
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ks, silhouette_scores, marker="o", linewidth=2, color="tab:orange")
        ax.set_title("Pontuação de Silhueta vs Número de Grupos")
        ax.set_xlabel("Número de grupos (k)")
        ax.set_ylabel("Silhueta")
        ax.grid(True)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster in sorted(df["Cluster"].unique()):
            mask = df["Cluster"] == cluster
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=40,
                alpha=0.75,
                label=f"Cluster {cluster}",
            )
        ax.set_title("Clusterização KMeans (PCA 2D)")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        summary["income_mean"].plot(kind="bar", ax=ax, color="tab:green")
        ax.set_title("Renda Média por Grupo")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Renda média")
        ax.grid(axis="y")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        summary["count"].plot(kind="bar", ax=ax, color="tab:purple")
        ax.set_title("Tamanho de cada grupo")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Quantidade de registros")
        ax.grid(axis="y")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        table_data = summary[["count", "income_mean", "age_mean", "debt_mean", "credit_score_mean", "approved_rate"]].reset_index()
        table = ax.table(
            cellText=table_data.values,
            colLabels=["Cluster", "Count", "Income Avg", "Age Avg", "Debt Avg", "Score Avg", "Approved Rate"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title("Resumo numérico por cluster")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        y = 0.94
        line_height = 0.055
        for cluster in sorted(df["Cluster"].unique()):
            group = df[df["Cluster"] == cluster]
            top_industries = group["Industry"].value_counts().head(3)
            top_ethnicity = group["Ethnicity"].value_counts().head(3)
            top_citizen = group["Citizen"].value_counts().head(3)
            top_zip = group["ZipCode"].value_counts().head(3)

            ax.text(0.01, y, f"Cluster {cluster}", fontsize=12, fontweight="bold")
            y -= line_height
            ax.text(
                0.02,
                y,
                "Top Indústrias: " + ", ".join([f"{idx} ({val})" for idx, val in top_industries.items()]),
                fontsize=10,
                wrap=True,
            )
            y -= line_height
            ax.text(
                0.02,
                y,
                "Top Etnias: " + ", ".join([f"{idx} ({val})" for idx, val in top_ethnicity.items()]),
                fontsize=10,
                wrap=True,
            )
            y -= line_height
            ax.text(
                0.02,
                y,
                "Top Citizenship: " + ", ".join([f"{idx} ({val})" for idx, val in top_citizen.items()]),
                fontsize=10,
                wrap=True,
            )
            y -= line_height
            ax.text(
                0.02,
                y,
                "Top CEPs: " + ", ".join([f"{int(idx)} ({val})" for idx, val in top_zip.items()]),
                fontsize=10,
                wrap=True,
            )
            y -= line_height * 1.5

        ax.axis("off")
        ax.set_title("Principais categorias por cluster")
        pdf.savefig(fig)
        plt.close(fig)

        conclusion = (
            f"Conclusão da análise de clusterização com KMeans\n\n"
            f"1. Melhor número de grupos: {best_k}. "
            "O critério foi baseado na pontuação de silhueta e na inércia do algoritmo.\n\n"
            "2. Características de cada grupo: o relatório descreve os clusters por "
            "indústria, etnia, citizen (tipo de cidadania) e CEP. Esses fatores foram usados para "
            "identificar perfis de clientes no dataset de créditos.\n\n"
            "3. Renda média de cada grupo: o gráfico e a tabela apresentam a renda média por cluster, "
            "permitindo comparar os grupos formados.\n\n"
            "4. Artefatos gerados: o script salvou o relatório PDF, uma atribuição de cluster por registro, "
            "um resumo numérico por cluster e uma descrição de categorias principais em CSV."
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis("off")
        wrapped = textwrap.fill(conclusion, width=90)
        ax.text(0.01, 0.95, wrapped, fontsize=11, va="top")
        ax.set_title("Conclusão da análise e artefatos gerados")
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path


def describe_groups(df: pd.DataFrame) -> pd.DataFrame:
    description = []
    for cluster, group in df.groupby("Cluster"):
        top_industries = group["Industry"].value_counts().head(3).to_dict()
        top_ethnicity = group["Ethnicity"].value_counts().head(3).to_dict()
        top_citizen = group["Citizen"].value_counts().head(3).to_dict()
        top_zip = group["ZipCode"].value_counts().head(3).to_dict()
        description.append(
            {
                "Cluster": cluster,
                "Count": len(group),
                "Average income": round(group["Income"].mean(), 2),
                "Average age": round(group["Age"].mean(), 2),
                "Average debt": round(group["Debt"].mean(), 2),
                "Average credit score": round(group["CreditScore"].mean(), 2),
                "Top industries": ", ".join([f"{k} ({v})" for k, v in top_industries.items()]),
                "Top ethnicities": ", ".join([f"{k} ({v})" for k, v in top_ethnicity.items()]),
                "Top citizenship": ", ".join([f"{k} ({v})" for k, v in top_citizen.items()]),
                "Top zipcodes": ", ".join([f"{int(k)} ({v})" for k, v in top_zip.items()]),
                "ZIP median": int(group["ZipCode"].median()),
            }
        )
    return pd.DataFrame(description)


def run(data_path: Path | None = None):
    default_data_path, artifacts_dir = get_paths()
    data_path = default_data_path if data_path is None else data_path

    print(f"Aluno: {STUDENT_NAME}")
    print(f"Carregando dados de: {data_path}")

    df = load_data(data_path)
    preprocessor = build_preprocessor(df)
    X = prepare_features(df, preprocessor)

    ks, inertia, silhouette_scores, best_k = find_best_k(X, k_min=2, k_max=10)
    print(f"Melhor número de grupos encontrado: {best_k}")

    model, labels = fit_kmeans(X, best_k)
    df["Cluster"] = labels

    summary = summarize_clusters(df)
    description = describe_groups(df)

    save_reports(df, summary, artifacts_dir)
    pdf_path = create_pdf_report(
        artifacts_dir,
        ks,
        inertia,
        silhouette_scores,
        best_k,
        summary,
        df,
        labels,
        X,
    )

    description.to_csv(artifacts_dir / "credito_cluster_description.csv", index=False)

    print("\nResumo de resultados:")
    print(summary)
    print("\nDescrição dos grupos no arquivo:")
    print(description)
    print(f"\nRelatório em PDF salvo em: {pdf_path}")
    print(f"Arquivos de saída salvos em: {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa análise de clusterização KMeans no dataset de crédito.")
    parser.add_argument(
        "--data",
        default=None,
        help="Caminho para o arquivo de dados de crédito (padrão: credito.csv).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Nome completo do aluno para incluir no relatório.",
    )
    args = parser.parse_args()

    if args.name:
        STUDENT_NAME = args.name

    run(Path(args.data).resolve() if args.data else None)
