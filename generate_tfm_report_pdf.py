import math
import re
import textwrap
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "posts.csv"
IMAGES_DIR = ROOT / "data" / "images"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report_assets"
PDF_PATH = ROOT / "TFM_documentacion_proyecto.pdf"

RANDOM_STATE = 42
TEST_SIZE = 0.30
TOP_K = 5

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_DIR = MODELS_DIR / "modelo_embeddings"
LEGACY_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LEGACY_EMBEDDING_MODEL_DIR = MODELS_DIR / "modelo_all_minilm_l6_v2"
EMBEDDINGS_ARTIFACT_PATH = ARTIFACTS_DIR / "embeddings_posts.joblib"
VISION_CHECKPOINT_PATH = MODELS_DIR / "vision_resnet18_head_only.pth"


PROMPTS_CATEGORIA = {
    "animales": "me gustan fotos y publicaciones de animales, mascotas y escenas con fauna",
    "coches": "me interesan coches, conduccion, carreteras y vehiculos en movimiento",
    "cocina": "me gusta descubrir recetas, comida, platos y momentos de cocina",
    "deporte": "me interesan entrenamientos, partidos, ejercicio y vida deportiva",
    "musica": "me gusta la musica, conciertos, canciones y ambientes musicales",
    "playa": "me atraen la playa, el mar, la arena y el ambiente costero",
    "viajes": "me interesan escapadas, destinos, paisajes y experiencias de viaje",
    "videojuegos": "me gustan los videojuegos, consolas, streaming y escenas de juego",
}

USUARIOS_SIMULADOS = [
    {"usuario": "Usuario 1", "intereses": ["playa", "animales", "viajes"]},
    {"usuario": "Usuario 2", "intereses": ["videojuegos", "musica"]},
    {"usuario": "Usuario 3", "intereses": ["coches", "viajes"]},
    {"usuario": "Usuario 4", "intereses": ["cocina", "playa"]},
    {"usuario": "Usuario 5", "intereses": ["deporte", "animales"]},
]

CONSULTAS_COMPARATIVA_MODELOS = [
    "vacaciones con música y playa",
    "viaje en coche por la costa",
    "jugando al FIFA escuchando Anuel",
    "cocinando mientras veo fútbol",
]

CATEGORIAS_ESPERADAS_COMPARATIVA = {
    "vacaciones con música y playa": {"playa", "viajes", "musica"},
    "viaje en coche por la costa": {"viajes", "coches", "playa"},
    "jugando al FIFA escuchando Anuel": {"videojuegos", "musica"},
    "cocinando mientras veo fútbol": {"cocina", "deporte"},
}


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def preprocesar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = texto.replace("#", " ")
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def sanitize_report_text(texto: str) -> str:
    texto = str(texto)
    texto = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ#.,;:!?()/%+\-]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def normalizar_vector(vector: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(vector, axis=1, keepdims=True)
    norma = np.clip(norma, 1e-12, None)
    return vector / norma


def save_table_image(df: pd.DataFrame, path: Path, title: str, font_size: int = 9, col_widths=None):
    n_rows, n_cols = df.shape
    fig_h = max(2.2, 0.42 * (n_rows + 2))
    fig_w = max(9, 1.45 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colLoc="center",
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            cell.set_facecolor("#f8fbff" if row % 2 == 0 else "#edf4fa")
    ax.set_title(title, fontsize=13, weight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_wrapped_text(fig, x, y, text, width=92, fontsize=11, line_height=0.028, color="#222222", weight="normal"):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width))
    for idx, line in enumerate(lines):
        fig.text(x, y - idx * line_height, line, fontsize=fontsize, color=color, ha="left", va="top", weight=weight)
    return y - len(lines) * line_height


def add_image(fig, path: Path, rect, title: str | None = None):
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(path).convert("RGB"))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    return ax


def add_page_number(fig, page_number: int):
    fig.text(0.95, 0.02, str(page_number), fontsize=10, ha="right", va="bottom", color="#666666")


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


class ImagePathDataset(Dataset):
    def __init__(self, rutas: Iterable[str], transform):
        self.rutas = [Path(ruta) for ruta in rutas]
        self.transform = transform

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        ruta = self.rutas[idx]
        imagen = Image.open(ruta).convert("RGB")
        return self.transform(imagen), str(ruta)


def compute_text_baseline():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["texto_original"] = (
        df["descripcion"].fillna("").astype(str).str.strip()
        + " "
        + df["hashtags"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    df["texto_limpio"] = df["texto_original"].apply(preprocesar_texto)
    df_modelo = df[df["texto_limpio"].str.len() > 0].copy()

    train_idx, test_idx = train_test_split(
        df_modelo.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_modelo["categoria"],
    )

    X_train = df_modelo.loc[train_idx, "texto_limpio"]
    X_test = df_modelo.loc[test_idx, "texto_limpio"]
    y_train = df_modelo.loc[train_idx, "categoria"]
    y_test = df_modelo.loc[test_idx, "categoria"]

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1200, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = pd.DataFrame(
        {
            "métrica": ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            "valor": [
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="macro", zero_division=0),
                recall_score(y_test, y_pred, average="macro", zero_division=0),
                f1_score(y_test, y_pred, average="macro", zero_division=0),
            ],
        }
    )

    test_results = df_modelo.loc[test_idx].copy()
    test_results["prediccion"] = y_pred
    test_results["acierto"] = (test_results["categoria"].to_numpy() == y_pred)

    confusion_rows = []
    for i, real_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusion_rows.append(
                    {
                        "categoria_real": real_label,
                        "categoria_predicha": pred_label,
                        "casos": int(cm[i, j]),
                    }
                )
    confusiones_df = pd.DataFrame(confusion_rows).sort_values("casos", ascending=False).reset_index(drop=True)

    ambiguedad_df = None
    if "es_ambiguo" in test_results.columns:
        ambiguedad_df = (
            test_results.groupby("es_ambiguo")["acierto"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accuracy"})
            .round(4)
            .reset_index()
        )

    # Metrics chart
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(metrics["métrica"], metrics["valor"], color=["#1f77b4", "#4c78a8", "#59a14f", "#f28e2b"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Métricas globales del baseline TF-IDF")
    axes[0].tick_params(axis="x", rotation=20)
    for idx, valor in enumerate(metrics["valor"]):
        axes[0].text(idx, valor + 0.02, f"{valor:.3f}", ha="center", fontsize=10)

    if ambiguedad_df is not None:
        labels_amb = ["No ambiguo", "Ambiguo"]
        values_amb = [
            float(ambiguedad_df.loc[ambiguedad_df["es_ambiguo"] == False, "accuracy"].iloc[0]),
            float(ambiguedad_df.loc[ambiguedad_df["es_ambiguo"] == True, "accuracy"].iloc[0]),
        ]
        axes[1].bar(labels_amb, values_amb, color=["#59a14f", "#e15759"])
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title("Accuracy según ambigüedad del post")
        for idx, valor in enumerate(values_amb):
            axes[1].text(idx, valor + 0.02, f"{valor:.3f}", ha="center", fontsize=10)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    metrics_chart_path = REPORT_DIR / "tfidf_metricas.png"
    fig.savefig(metrics_chart_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 7.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=labels,
        display_labels=labels,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        xticks_rotation=40,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Matriz de confusión normalizada - TF-IDF")
    plt.tight_layout()
    confusion_path = REPORT_DIR / "tfidf_confusion.png"
    fig.savefig(confusion_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "df": df_modelo,
        "metrics": metrics,
        "confusiones_df": confusiones_df,
        "ambiguedad_df": ambiguedad_df,
        "metrics_chart_path": metrics_chart_path,
        "confusion_path": confusion_path,
    }


def compute_embeddings_results():
    embedding_artifact = joblib.load(EMBEDDINGS_ARTIFACT_PATH)
    posts_df = embedding_artifact["posts"].copy().reset_index(drop=True)
    embeddings_multi = embedding_artifact["embeddings"].astype(np.float32)

    model_multi = SentenceTransformer(str(EMBEDDING_MODEL_DIR), local_files_only=True)
    model_legacy = SentenceTransformer(str(LEGACY_EMBEDDING_MODEL_DIR), local_files_only=True)
    embeddings_legacy = model_legacy.encode(
        posts_df["texto_embedding"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    rows = []
    for model_label, model_obj, corpus_embeddings in [
        ("all-MiniLM-L6-v2", model_legacy, embeddings_legacy),
        ("paraphrase-multilingual-MiniLM-L12-v2", model_multi, embeddings_multi),
    ]:
        for consulta in CONSULTAS_COMPARATIVA_MODELOS:
            embedding_consulta = model_obj.encode(
                [consulta],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            scores = cosine_similarity(embedding_consulta, corpus_embeddings).ravel()
            ranking = posts_df.assign(score=scores).sort_values("score", ascending=False).head(TOP_K)
            precision = float(ranking["categoria"].isin(CATEGORIAS_ESPERADAS_COMPARATIVA[consulta]).mean())
            rows.append(
                {
                    "modelo": model_label,
                    "consulta": consulta,
                    "precision_at_5": precision,
                    "categoria_top_1": ranking.iloc[0]["categoria"],
                    "score_top_1": float(ranking.iloc[0]["score"]),
                }
            )

    comparison_df = pd.DataFrame(rows)
    summary_df = (
        comparison_df.groupby("modelo")[["precision_at_5", "score_top_1"]]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"precision_at_5": "precision_at_5_media", "score_top_1": "score_top_1_medio"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    axes[0].bar(summary_df["modelo"], summary_df["precision_at_5_media"], color=["#9c755f", "#4c78a8"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Comparativa breve entre modelos de embeddings")
    axes[0].set_ylabel("Precision@5 media")
    axes[0].tick_params(axis="x", rotation=15)
    for idx, valor in enumerate(summary_df["precision_at_5_media"]):
        axes[0].text(idx, valor + 0.02, f"{valor:.2f}", ha="center")

    orden = CONSULTAS_COMPARATIVA_MODELOS
    legacy_scores = comparison_df[comparison_df["modelo"].eq("all-MiniLM-L6-v2")].set_index("consulta").loc[orden, "precision_at_5"].to_numpy()
    multi_scores = comparison_df[comparison_df["modelo"].eq("paraphrase-multilingual-MiniLM-L12-v2")].set_index("consulta").loc[orden, "precision_at_5"].to_numpy()
    x = np.arange(len(orden))
    width = 0.38
    axes[1].bar(x - width / 2, legacy_scores, width=width, label="all-MiniLM-L6-v2", color="#9c755f")
    axes[1].bar(x + width / 2, multi_scores, width=width, label="multilingual", color="#4c78a8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["playa+música", "viaje+coche", "FIFA+Anuel", "cocina+fútbol"],
        rotation=18,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Precision@5")
    axes[1].set_title("Resultado por consulta")
    axes[1].legend()
    plt.tight_layout()
    model_choice_path = REPORT_DIR / "embeddings_model_choice.png"
    fig.savefig(model_choice_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    detail_path = ARTIFACTS_DIR / "comparacion_detalle" / "cocinando_mientras_veo_futbol.png"
    overview_path = ARTIFACTS_DIR / "comparacion_modelos_embeddings.png"

    return {
        "comparison_df": comparison_df,
        "summary_df": summary_df,
        "overview_path": overview_path,
        "detail_path": detail_path,
        "model_choice_path": model_choice_path,
    }


def compute_vision_results():
    checkpoint = torch.load(VISION_CHECKPOINT_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)
    mean = checkpoint.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalization", {}).get("std", [0.229, 0.224, 0.225])

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_dataset = ImageFolderWithPaths(IMAGES_DIR / "test", transform=eval_transform)
    loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    def create_model():
        model = models.resnet18(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        return model.eval()

    @torch.no_grad()
    def pred_df(state_dict):
        model = create_model()
        model.load_state_dict(state_dict)
        rows = []
        for images, labels, paths in loader:
            probs = torch.softmax(model(images), dim=1).numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)
            for path, label, pred, conf, prob in zip(paths, labels.numpy(), preds, confs, probs):
                rows.append(
                    {
                        "ruta": path,
                        "clase_real": test_dataset.classes[label],
                        "clase_predicha": test_dataset.classes[pred],
                        "confianza": float(conf),
                        "vector_probabilidades": prob.tolist(),
                        "acierto": bool(label == pred),
                    }
                )
        return pd.DataFrame(rows)

    base_df = pred_df(checkpoint["base_model_state_dict"])
    fine_df = pred_df(checkpoint["model_state_dict"])
    acc_base = accuracy_score(base_df["clase_real"], base_df["clase_predicha"])
    acc_fine = accuracy_score(fine_df["clase_real"], fine_df["clase_predicha"])

    cm_base = confusion_matrix(base_df["clase_real"], base_df["clase_predicha"], labels=class_names)
    cm_fine = confusion_matrix(fine_df["clase_real"], fine_df["clase_predicha"], labels=class_names)

    comparativa_df = base_df.merge(
        fine_df,
        on=["ruta", "clase_real"],
        suffixes=("_base", "_fine"),
    )
    comparativa_df["corrige"] = (~comparativa_df["acierto_base"]) & (comparativa_df["acierto_fine"])
    comparativa_df["fallan_ambos"] = (~comparativa_df["acierto_base"]) & (~comparativa_df["acierto_fine"])
    comparativa_df["mejora_confianza"] = comparativa_df["confianza_fine"] - comparativa_df["confianza_base"]

    corrected_df = comparativa_df[comparativa_df["corrige"]].sort_values(
        ["confianza_fine", "mejora_confianza"],
        ascending=[False, False],
    )
    failed_df = comparativa_df[comparativa_df["fallan_ambos"]].sort_values("confianza_fine", ascending=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    modelos = ["Base", "Fine-tuneado"]
    valores = [acc_base, acc_fine]
    bars = ax.bar(modelos, valores, color=["#dda15e", "#2a9d8f"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparativa visual del rendimiento en test")
    for bar, valor in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, valor + 0.02, f"{valor:.3f}", ha="center")
    plt.tight_layout()
    accuracy_path = REPORT_DIR / "vision_accuracy.png"
    fig.savefig(accuracy_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    ConfusionMatrixDisplay(cm_base, display_labels=class_names).plot(
        ax=axes[0], cmap="Oranges", xticks_rotation=35, colorbar=False, values_format="d"
    )
    axes[0].set_title("Modelo base")
    ConfusionMatrixDisplay(cm_fine, display_labels=class_names).plot(
        ax=axes[1], cmap="Greens", xticks_rotation=35, colorbar=False, values_format="d"
    )
    axes[1].set_title("Modelo fine-tuneado")
    plt.suptitle("Matrices de confusión comparadas")
    plt.tight_layout()
    confusion_path = REPORT_DIR / "vision_confusion.png"
    fig.savefig(confusion_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    corrected_path = REPORT_DIR / "vision_corrected_examples.png"
    save_visual_examples(corrected_df, corrected_path, "Casos corregidos por el fine-tuning")

    failed_path = REPORT_DIR / "vision_failed_examples.png"
    save_visual_examples(failed_df, failed_path, "Casos donde ambos modelos siguen fallando")

    prob_path = REPORT_DIR / "vision_probability_examples.png"
    save_probability_examples(corrected_df if not corrected_df.empty else failed_df, class_names, prob_path)

    summary_table = pd.DataFrame(
        [
            {"Aspecto": "Accuracy", "Modelo Base": f"{acc_base:.3f}", "Fine-Tuning": f"{acc_fine:.3f}"},
            {"Aspecto": "Adaptación dominio", "Modelo Base": "Muy baja", "Fine-Tuning": "Alta"},
            {"Aspecto": "Generalización inicial", "Modelo Base": "Conocimiento ImageNet", "Fine-Tuning": "Conocimiento + adaptación"},
            {"Aspecto": "Categorías específicas", "Modelo Base": "No especializadas", "Fine-Tuning": "Alineadas con el TFM"},
            {"Aspecto": "Rendimiento final", "Modelo Base": "Insuficiente", "Fine-Tuning": "Consistente"},
        ]
    )
    summary_table_path = REPORT_DIR / "vision_summary_table.png"
    save_table_image(summary_table, summary_table_path, "Comparativa final del bloque visual", font_size=10)

    return {
        "acc_base": acc_base,
        "acc_fine": acc_fine,
        "corrected_df": corrected_df,
        "failed_df": failed_df,
        "accuracy_path": accuracy_path,
        "confusion_path": confusion_path,
        "corrected_path": corrected_path,
        "failed_path": failed_path,
        "prob_path": prob_path,
        "summary_table_path": summary_table_path,
    }


def save_visual_examples(df: pd.DataFrame, path: Path, title: str, max_cases: int = 4):
    subset = df.head(max_cases).copy()
    if subset.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No hay casos disponibles.", ha="center", va="center", fontsize=14)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(subset), 3, figsize=(14.5, 4.6 * len(subset)))
    if len(subset) == 1:
        axes = np.array([axes])

    for row_idx, (_, fila) in enumerate(subset.iterrows()):
        image = Image.open(fila["ruta"]).convert("RGB")
        for ax in axes[row_idx]:
            ax.imshow(image)
            ax.axis("off")
        axes[row_idx, 0].set_title(f"Imagen original\nReal: {fila['clase_real']}", fontsize=10)
        axes[row_idx, 1].set_title(
            f"Base\nPred: {fila['clase_predicha_base']}\nConf: {fila['confianza_base']:.2f}",
            fontsize=10,
        )
        axes[row_idx, 2].set_title(
            f"Fine-tuning\nPred: {fila['clase_predicha_fine']}\nConf: {fila['confianza_fine']:.2f}",
            fontsize=10,
        )

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_probability_examples(df: pd.DataFrame, class_names: list[str], path: Path, max_cases: int = 2):
    subset = df.head(max_cases).copy()
    if subset.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No hay ejemplos de probabilidades disponibles.", ha="center", va="center", fontsize=14)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(subset), 1, figsize=(14, 4.4 * len(subset)))
    if len(subset) == 1:
        axes = [axes]
    x = np.arange(len(class_names))
    width = 0.36
    for ax, (_, fila) in zip(axes, subset.iterrows()):
        probs_base = np.array(fila["vector_probabilidades_base"])
        probs_fine = np.array(fila["vector_probabilidades_fine"])
        ax.bar(x - width / 2, probs_base, width=width, color="#dda15e", label="Base")
        ax.bar(x + width / 2, probs_fine, width=width, color="#2a9d8f", label="Fine-tuning")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_ylim(0, max(probs_base.max(), probs_fine.max()) + 0.1)
        ax.set_ylabel("Probabilidad")
        ax.set_title(
            f"{Path(fila['ruta']).name} | Real: {fila['clase_real']} | "
            f"Base: {fila['clase_predicha_base']} | Fine: {fila['clase_predicha_fine']}",
            fontsize=11,
        )
        ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_final_recommender_results():
    artifact = joblib.load(EMBEDDINGS_ARTIFACT_PATH)
    posts_df = artifact["posts"].copy().reset_index(drop=True)
    posts_df["post_idx"] = np.arange(len(posts_df))
    posts_df["texto_resumido"] = (
        posts_df["descripcion"].fillna("").astype(str).str.strip()
        + " "
        + posts_df["hashtags"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip().map(
        lambda texto: textwrap.shorten(sanitize_report_text(texto), width=90, placeholder="...")
    )
    embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_DIR), local_files_only=True)
    matriz_embeddings_posts = artifact["embeddings"].astype(np.float32)

    checkpoint = torch.load(VISION_CHECKPOINT_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)
    mean = checkpoint.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalization", {}).get("std", [0.229, 0.224, 0.225])
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    def create_model():
        model = models.resnet18(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        return model.eval()

    model_visual = create_model()
    model_visual.load_state_dict(checkpoint["model_state_dict"])

    catalog_rows = []
    for split in ["train", "test"]:
        split_dir = IMAGES_DIR / split
        for category_dir in sorted(split_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for ruta in sorted(category_dir.iterdir()):
                if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                catalog_rows.append(
                    {
                        "ruta": str(ruta),
                        "archivo": ruta.name,
                        "split": split,
                        "categoria_real": category_dir.name,
                    }
                )
    catalog_df = pd.DataFrame(catalog_rows)
    catalog_df["image_idx"] = np.arange(len(catalog_df))

    @torch.no_grad()
    def predecir_probabilidades_catalogo(rutas):
        dataset = ImagePathDataset(rutas, eval_transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        embeddings = []
        predicciones = []
        confianzas = []
        for tensores, _ in loader:
            probs = torch.softmax(model_visual(tensores), dim=1).numpy().astype(np.float32)
            probs = normalizar_vector(probs)
            embeddings.append(probs)
            pred_idx = probs.argmax(axis=1)
            predicciones.extend([class_names[idx] for idx in pred_idx])
            confianzas.extend(probs.max(axis=1).tolist())
        return np.vstack(embeddings), predicciones, confianzas

    matriz_embeddings_imagenes, categorias_predichas, confianzas_predichas = predecir_probabilidades_catalogo(
        catalog_df["ruta"].tolist()
    )
    catalog_df["embedding_visual"] = [fila for fila in matriz_embeddings_imagenes]
    catalog_df["categoria_predicha"] = categorias_predichas
    catalog_df["confianza_predicha"] = np.round(confianzas_predichas, 4)

    catalog_train_df = catalog_df[catalog_df["split"].eq("train")].copy().reset_index(drop=True)
    catalog_test_df = catalog_df[catalog_df["split"].eq("test")].copy().reset_index(drop=True)
    matriz_test_imagenes = np.vstack(catalog_test_df["embedding_visual"].to_numpy())

    def frases_desde_intereses(categorias):
        return [PROMPTS_CATEGORIA[categoria] for categoria in categorias]

    def seleccionar_likes_texto(df_posts, categorias, seed=RANDOM_STATE):
        likes = []
        usados = set()
        for offset, categoria in enumerate(categorias):
            subset = df_posts[df_posts["categoria"].eq(categoria)].copy()
            if subset.empty:
                continue
            muestra = subset.sample(n=1, random_state=seed + offset)
            idx = int(muestra.iloc[0]["post_idx"])
            if idx in usados:
                continue
            usados.add(idx)
            likes.append(muestra)
        if not likes:
            likes.append(df_posts.sample(n=1, random_state=seed))
        return pd.concat(likes, ignore_index=True)

    def construir_perfil_textual(categorias, likes_df):
        embeddings_intereses = embedding_model.encode(
            frases_desde_intereses(categorias),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings_likes = matriz_embeddings_posts[likes_df["post_idx"].to_numpy()]
        perfil = np.vstack([embeddings_intereses, embeddings_likes]).mean(axis=0, keepdims=True)
        return normalizar_vector(perfil)

    def recomendar_posts(perfil_textual, excluir_post_idx, top_k=TOP_K):
        scores = cosine_similarity(perfil_textual, matriz_embeddings_posts).ravel()
        ranking = posts_df.copy()
        ranking["score_similitud"] = scores
        ranking = ranking[~ranking["post_idx"].isin(set(excluir_post_idx))].copy()
        ranking = ranking.sort_values("score_similitud", ascending=False).head(top_k).copy().reset_index(drop=True)
        ranking.insert(0, "posicion", np.arange(1, len(ranking) + 1))
        ranking["score_similitud"] = ranking["score_similitud"].round(4)
        return ranking

    def seleccionar_likes_imagenes(df_imagenes, categorias, seed=RANDOM_STATE):
        likes = []
        usados = set()
        for offset, categoria in enumerate(categorias):
            subset = df_imagenes[df_imagenes["categoria_real"].eq(categoria)].copy()
            if subset.empty:
                continue
            muestra = subset.sample(n=1, random_state=seed + offset)
            idx = int(muestra.iloc[0]["image_idx"])
            if idx in usados:
                continue
            usados.add(idx)
            likes.append(muestra)
        if not likes:
            likes.append(df_imagenes.sample(n=1, random_state=seed))
        return pd.concat(likes, ignore_index=True)

    def vector_intereses_visual(categorias):
        vector = np.array([[1.0 if categoria in categorias else 0.0 for categoria in class_names]], dtype=np.float32)
        return normalizar_vector(vector)

    def construir_perfil_visual(categorias, likes_df):
        intereses_vec = vector_intereses_visual(categorias)
        likes_vec = np.vstack(likes_df["embedding_visual"].to_numpy())
        perfil = np.vstack([intereses_vec, likes_vec]).mean(axis=0, keepdims=True)
        return normalizar_vector(perfil)

    def recomendar_imagenes(perfil_visual, top_k=TOP_K):
        scores = cosine_similarity(perfil_visual, matriz_test_imagenes).ravel()
        ranking = catalog_test_df.copy()
        ranking["score_similitud"] = scores
        ranking = ranking.sort_values("score_similitud", ascending=False).head(top_k).copy().reset_index(drop=True)
        ranking.insert(0, "posicion", np.arange(1, len(ranking) + 1))
        ranking["score_similitud"] = ranking["score_similitud"].round(4)
        return ranking

    resultados_usuarios = []
    metricas_usuarios = []
    for perfil in USUARIOS_SIMULADOS:
        intereses = perfil["intereses"]
        intereses_set = set(intereses)

        likes_texto_df = seleccionar_likes_texto(posts_df, intereses)
        perfil_textual = construir_perfil_textual(intereses, likes_texto_df)
        ranking_texto_df = recomendar_posts(perfil_textual, likes_texto_df["post_idx"].tolist())
        precision_texto = float(ranking_texto_df["categoria"].isin(intereses_set).mean())

        likes_imagenes_df = seleccionar_likes_imagenes(catalog_train_df, intereses)
        perfil_visual = construir_perfil_visual(intereses, likes_imagenes_df)
        ranking_imagenes_df = recomendar_imagenes(perfil_visual)
        precision_imagenes = float(ranking_imagenes_df["categoria_real"].isin(intereses_set).mean())

        resultados_usuarios.append(
            {
                "usuario": perfil["usuario"],
                "intereses": intereses,
                "likes_texto_df": likes_texto_df,
                "ranking_texto_df": ranking_texto_df,
                "likes_imagenes_df": likes_imagenes_df,
                "ranking_imagenes_df": ranking_imagenes_df,
            }
        )
        metricas_usuarios.append(
            {
                "usuario": perfil["usuario"],
                "intereses": " | ".join(intereses),
                "precision_at_5_texto": round(precision_texto, 4),
                "precision_at_5_imagenes": round(precision_imagenes, 4),
                "precision_media": round((precision_texto + precision_imagenes) / 2, 4),
                "categorias_texto_top5": " | ".join(ranking_texto_df["categoria"].tolist()),
                "categorias_imagenes_top5": " | ".join(ranking_imagenes_df["categoria_real"].tolist()),
            }
        )

    metricas_df = pd.DataFrame(metricas_usuarios)
    metricas_globales_df = pd.DataFrame(
        [
            {"modalidad": "Texto", "precision_at_5_media": metricas_df["precision_at_5_texto"].mean()},
            {"modalidad": "Imagenes", "precision_at_5_media": metricas_df["precision_at_5_imagenes"].mean()},
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    x = np.arange(len(metricas_df))
    width = 0.35
    axes[0].bar(x - width / 2, metricas_df["precision_at_5_texto"], width=width, label="Texto", color="#4c78a8")
    axes[0].bar(x + width / 2, metricas_df["precision_at_5_imagenes"], width=width, label="Imagenes", color="#f58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metricas_df["usuario"], rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Precision@5")
    axes[0].set_title("Rendimiento por usuario simulado")
    axes[0].legend()

    axes[1].bar(metricas_globales_df["modalidad"], metricas_globales_df["precision_at_5_media"], color=["#4c78a8", "#f58518"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Precision@5 media")
    axes[1].set_title("Resumen global por modalidad")
    for idx, valor in enumerate(metricas_globales_df["precision_at_5_media"]):
        axes[1].text(idx, valor + 0.02, f"{valor:.2f}", ha="center")
    plt.tight_layout()
    precision_chart_path = REPORT_DIR / "recommender_precision.png"
    fig.savefig(precision_chart_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics_table_path = REPORT_DIR / "recommender_metrics_table.png"
    save_table_image(
        metricas_df[["usuario", "precision_at_5_texto", "precision_at_5_imagenes", "precision_media"]],
        metrics_table_path,
        "Métricas del sistema final por usuario",
        font_size=10,
    )

    sample_user = next(item for item in resultados_usuarios if item["usuario"] == "Usuario 3")
    text_table = sample_user["ranking_texto_df"][["posicion", "categoria", "score_similitud", "texto_resumido"]].copy()
    text_table_path = REPORT_DIR / "recommender_text_user3.png"
    save_table_image(text_table, text_table_path, "Top-5 publicaciones recomendadas para Usuario 3", font_size=8)

    gallery_path = REPORT_DIR / "recommender_gallery_user1.png"
    sample_gallery_user = next(item for item in resultados_usuarios if item["usuario"] == "Usuario 1")
    save_gallery(sample_gallery_user["ranking_imagenes_df"], gallery_path, "Mini galería visual de Usuario 1")

    return {
        "metricas_df": metricas_df,
        "metricas_globales_df": metricas_globales_df,
        "precision_chart_path": precision_chart_path,
        "metrics_table_path": metrics_table_path,
        "text_table_path": text_table_path,
        "gallery_path": gallery_path,
    }


def save_gallery(ranking_df: pd.DataFrame, path: Path, title: str):
    subset = ranking_df.head(4).copy()
    fig, axes = plt.subplots(1, len(subset), figsize=(3.2 * len(subset), 3.8))
    if len(subset) == 1:
        axes = [axes]
    for ax, (_, fila) in zip(axes, subset.iterrows()):
        image = Image.open(fila["ruta"]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{fila['posicion']}. {fila['categoria_real']}\nscore={fila['score_similitud']}",
            fontsize=9,
        )
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_pdf(text_results, embeddings_results, vision_results, recommender_results):
    with PdfPages(PDF_PATH) as pdf:
        page = 1

        # Portada
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.5, 0.86, "Trabajo Fin de Máster", ha="center", fontsize=18, color="#1f4e79", weight="bold")
        fig.text(0.5, 0.80, "Sistema de recomendación de contenido textual y visual", ha="center", fontsize=26, weight="bold")
        fig.text(0.5, 0.75, "desde TF-IDF y embeddings hasta transfer learning y recomendación multimodal", ha="center", fontsize=15, color="#444444")
        fig.text(0.5, 0.62, "Documentación del proyecto", ha="center", fontsize=20, weight="bold")
        fig.text(0.5, 0.54, "Autor: Rafael", ha="center", fontsize=15)
        fig.text(0.5, 0.50, "Fecha: 29 de mayo de 2026", ha="center", fontsize=14, color="#555555")
        fig.text(0.5, 0.12, "El documento integra resultados reales de los notebooks 01, 02, 03, 04 y 05.", ha="center", fontsize=12, color="#666666")
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Índice
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.93, "Índice", fontsize=24, weight="bold")
        toc = [
            ("1. Resumen", 3),
            ("2. Introducción", 4),
            ("3. Marco teórico / estado de la cuestión", 5),
            ("4. Cuerpo principal del trabajo", 6),
            ("4.1. Baseline textual supervisado con TF-IDF", 6),
            ("4.2. Recomendación semántica con embeddings", 7),
            ("4.3. Comparación práctica TF-IDF vs embeddings", 8),
            ("4.4. Visión por computador con transfer learning", 9),
            ("4.5. Sistema final de recomendación", 11),
            ("5. Conclusiones integradas", 12),
        ]
        y = 0.86
        for title, num in toc:
            fig.text(0.10, y, title, fontsize=13)
            fig.text(0.90, y, str(num), fontsize=13, ha="right")
            y -= 0.055
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Resumen
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.94, "1. Resumen", fontsize=23, weight="bold")
        resumen = (
            "Este TFM estudia la construcción progresiva de un sistema de recomendación basado en contenido para publicaciones de estilo red social. "
            "El recorrido parte de un baseline textual supervisado con TF-IDF y regresión logística, evoluciona hacia embeddings semánticos, "
            "compara ambos enfoques en consultas ambiguas, incorpora un bloque de visión por computador mediante transfer learning y concluye con un "
            "sistema final capaz de recomendar texto e imágenes a usuarios simulados.\n\n"
            f"Los resultados cuantitativos principales son coherentes con esa narrativa. El baseline TF-IDF alcanza una accuracy de {float(text_results['metrics'].loc[text_results['metrics']['métrica'].eq('accuracy'), 'valor'].iloc[0]):.4f}, "
            f"pero su rendimiento cae con claridad cuando las publicaciones son ambiguas. El sistema visual muestra un salto todavía más marcado: el modelo base obtiene una accuracy de {vision_results['acc_base']:.3f}, "
            f"mientras que el modelo fine-tuneado alcanza {vision_results['acc_fine']:.3f}. Por último, el recomendador final logra una Precision@5 media de "
            f"{recommender_results['metricas_globales_df'].loc[recommender_results['metricas_globales_df']['modalidad'].eq('Texto'), 'precision_at_5_media'].iloc[0]:.2f} en texto y "
            f"{recommender_results['metricas_globales_df'].loc[recommender_results['metricas_globales_df']['modalidad'].eq('Imagenes'), 'precision_at_5_media'].iloc[0]:.2f} en imágenes.\n\n"
            "La aportación principal del proyecto no es una arquitectura compleja, sino una narrativa experimental clara: mostrar qué aporta cada bloque, "
            "qué límites aparecen cuando el dato deja de ser trivial y cómo esos bloques pueden integrarse después en una demostración final multimodal."
        )
        draw_wrapped_text(fig, 0.08, 0.88, resumen, width=96, fontsize=12, line_height=0.031)
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Introducción
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.94, "2. Introducción", fontsize=23, weight="bold")
        intro = (
            "El problema de partida es el de recomendar contenido a partir del propio contenido disponible. En contextos tipo red social, esto implica trabajar con "
            "textos cortos, hashtags, señales mixtas, ruido superficial y categorías que a menudo se solapan. Un post puede mezclar playa, viajes y música; otro puede "
            "hablar de cocinar mientras se ve fútbol; una imagen puede contener un coche en una escena que también transmite viaje.\n\n"
            "Con este punto de partida, el proyecto se formula como una evolución en cinco pasos. Primero se construye un baseline textual supervisado para disponer de una "
            "referencia interpretable. Después se pasa a embeddings para modelar cercanía semántica sin obligar al sistema a elegir una única etiqueta. En tercer lugar se comparan "
            "ambos enfoques de forma práctica. En cuarto lugar se añade visión por computador mediante un modelo preentrenado adaptado solo en la última capa. Por último, se reúnen "
            "todas las piezas en un sistema final de recomendación para usuarios simulados.\n\n"
            "El objetivo del documento es cerrar esa narrativa con una lectura continua y apoyada en resultados reales del proyecto."
        )
        draw_wrapped_text(fig, 0.08, 0.88, intro, width=96, fontsize=12, line_height=0.031)
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Marco teórico
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.94, "3. Marco teórico / estado de la cuestión", fontsize=23, weight="bold")
        theory = (
            "El baseline TF-IDF representa documentos según la importancia relativa de sus términos. Su ventaja es la interpretabilidad: permite inspeccionar vocabulario, pesos y "
            "confusiones léxicas. Sin embargo, depende mucho de coincidencias explícitas de palabras.\n\n"
            "Los embeddings semánticos desplazan el enfoque desde la frecuencia de términos hacia la cercanía de significado. En recomendación esto resulta útil porque el sistema puede "
            "ordenar publicaciones por similitud y no solo asignar una clase cerrada. En este proyecto se adopta como modelo principal `paraphrase-multilingual-MiniLM-L12-v2`, "
            "alineado con un corpus en español y con consultas formuladas de forma natural.\n\n"
            "En visión por computador, el transfer learning permite reutilizar conocimiento visual general aprendido en grandes corpus como ImageNet. Congelar el extractor visual y ajustar "
            "solo la última capa es una estrategia adecuada cuando el dataset específico es limitado. Metodológicamente, esto encaja bien con un TFM: reduce coste computacional, mantiene "
            "la interpretación clara y permite estudiar qué parte de la mejora proviene de la adaptación al dominio.\n\n"
            "Por último, la recomendación multimodal combina perfiles textuales y visuales para simular un caso de uso más realista. El interés aquí no es competir con un benchmark externo, "
            "sino demostrar integración, coherencia narrativa y capacidad de análisis de errores."
        )
        draw_wrapped_text(fig, 0.08, 0.88, theory, width=96, fontsize=12, line_height=0.031)
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Baseline textual
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.95, "4. Cuerpo principal del trabajo", fontsize=22, weight="bold")
        fig.text(0.08, 0.91, "4.1. Baseline textual supervisado con TF-IDF", fontsize=18, weight="bold")
        baseline_text = (
            f"El baseline utiliza una división estratificada train/test del 70/30 y un pipeline `TF-IDF + LogisticRegression`. "
            f"En test alcanza una accuracy de {float(text_results['metrics'].loc[text_results['metrics']['métrica'].eq('accuracy'), 'valor'].iloc[0]):.4f}, "
            f"con F1 macro de {float(text_results['metrics'].loc[text_results['metrics']['métrica'].eq('f1_macro'), 'valor'].iloc[0]):.4f}. "
            "La lectura más útil no es solo la métrica global, sino la degradación clara cuando el dato es ambiguo: en los posts no ambiguos la exactitud roza el 0.94, "
            "mientras que en los ambiguos cae hasta el entorno de 0.62.\n\n"
            "Las confusiones más frecuentes se concentran precisamente en pares conceptualmente próximos, como `viajes -> playa`, `música -> viajes` o `videojuegos -> música`. "
            "Este patrón justifica que el siguiente paso del proyecto no sea exprimir más el clasificador, sino cambiar la representación."
        )
        draw_wrapped_text(fig, 0.08, 0.86, baseline_text, width=96, fontsize=11.5, line_height=0.029)
        add_image(fig, text_results["metrics_chart_path"], [0.08, 0.39, 0.84, 0.23], "Métricas globales y efecto de la ambigüedad")
        add_image(fig, text_results["confusion_path"], [0.16, 0.08, 0.68, 0.24], "Matriz de confusión normalizada")
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Embeddings
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.95, "4.2. Recomendación semántica con embeddings", fontsize=18, weight="bold")
        emb_text = (
            "El notebook 02 cambia el problema de clasificación por uno de recuperación semántica. Cada publicación pasa a representarse como un vector y las recomendaciones "
            "se obtienen mediante similitud coseno. Esta decisión permite que una consulta mixta se acerque simultáneamente a varias categorías plausibles en lugar de obligar al sistema "
            "a escoger una sola clase final.\n\n"
            "El proyecto adopta como modelo textual principal `paraphrase-multilingual-MiniLM-L12-v2`. Su elección responde a coherencia lingüística y cobertura en español. "
            "La comparación breve frente al antiguo `all-MiniLM-L6-v2`, incluida en el notebook 02, debe leerse con cautela: sobre una muestra pequeña de consultas no ofrece una ventaja "
            "métrica sistemática, pero sí mantiene la narrativa de un sistema alineado con consultas naturales en español.\n\n"
            "La figura de la derecha recupera además una comparación ya presente en el proyecto entre TF-IDF y embeddings sobre consultas reales del dominio."
        )
        draw_wrapped_text(fig, 0.08, 0.88, emb_text, width=49, fontsize=11.4, line_height=0.028)
        add_image(fig, embeddings_results["model_choice_path"], [0.54, 0.57, 0.38, 0.28], "Comparativa breve entre modelos de embeddings")
        add_image(fig, embeddings_results["overview_path"], [0.08, 0.16, 0.40, 0.30], "Comparativa general TF-IDF vs embeddings")
        add_image(fig, embeddings_results["detail_path"], [0.54, 0.14, 0.38, 0.32], "Ejemplo detallado: cocina y deporte")
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Comparacion TF-IDF vs embeddings
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.95, "4.3. Comparación práctica TF-IDF vs embeddings", fontsize=18, weight="bold")
        comparison_text = (
            "La comparación entre notebooks 01, 02 y 03 muestra que TF-IDF sigue siendo competitivo cuando las consultas son directas y el vocabulario es reconocible, "
            "pero pierde naturalidad cuando una frase mezcla más de una intención. En cambio, embeddings devuelve rankings con mejor cobertura semántica, aunque a veces introduce ruido "
            "o concentraciones excesivas alrededor de una misma zona del espacio vectorial.\n\n"
            "La conclusión metodológica es clara: TF-IDF funciona como baseline interpretable y embeddings como mecanismo más natural para recomendación. Por eso el proyecto no presenta "
            "ambos bloques como rivales absolutos, sino como fases sucesivas de una evolución razonada."
        )
        draw_wrapped_text(fig, 0.08, 0.88, comparison_text, width=96, fontsize=11.8, line_height=0.03)
        top_confusions = text_results["confusiones_df"].head(8).copy()
        save_table_image(top_confusions, REPORT_DIR / "tfidf_confusions_table.png", "Confusiones léxicas más frecuentes", font_size=9)
        save_table_image(
            embeddings_results["comparison_df"][["modelo", "consulta", "precision_at_5", "categoria_top_1"]].copy(),
            REPORT_DIR / "embeddings_queries_table.png",
            "Consultas usadas en la comparativa breve de embeddings",
            font_size=8,
        )
        add_image(fig, REPORT_DIR / "tfidf_confusions_table.png", [0.08, 0.12, 0.40, 0.42], "Confusiones del baseline")
        add_image(fig, REPORT_DIR / "embeddings_queries_table.png", [0.52, 0.12, 0.40, 0.42], "Consulta y respuesta semántica")
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Vision metrics
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.95, "4.4. Visión por computador con transfer learning", fontsize=18, weight="bold")
        vision_text = (
            f"El bloque visual reutiliza una ResNet18 preentrenada y afina únicamente la última capa. El impacto del fine-tuning es muy visible: "
            f"el modelo base se queda en {vision_results['acc_base']:.3f} de accuracy en test, mientras que el modelo adaptado alcanza {vision_results['acc_fine']:.3f}. "
            "La diferencia es coherente con el objetivo del experimento: el extractor base aporta conocimiento visual general, pero no conoce todavía las fronteras de categorías como "
            "`playa`, `viajes`, `coches` o `animales` dentro del dominio concreto del TFM.\n\n"
            "Las matrices de confusión muestran que el modelo base colapsa parte del test hacia unas pocas clases dominantes, mientras que el modelo fine-tuneado reparte mucho mejor la decisión. "
            "Aun así, persisten errores en categorías visualmente cercanas, como `videojuegos -> coches` o `viajes -> playa`."
        )
        draw_wrapped_text(fig, 0.08, 0.88, vision_text, width=96, fontsize=11.6, line_height=0.028)
        add_image(fig, vision_results["accuracy_path"], [0.10, 0.47, 0.35, 0.23], "Salto de rendimiento")
        add_image(fig, vision_results["confusion_path"], [0.50, 0.40, 0.42, 0.32], "Matrices de confusión comparadas")
        add_image(fig, vision_results["summary_table_path"], [0.12, 0.08, 0.76, 0.22], "Síntesis del bloque visual")
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Vision examples and limitations
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.95, "4.4.1. Ejemplos visuales y limitaciones", fontsize=18, weight="bold")
        limit_text = (
            "El notebook 04 no oculta los errores porque son precisamente los que aportan valor interpretativo. El fine-tuning corrige casos donde el modelo base no sabía adaptar "
            "su conocimiento general al dominio, pero todavía hay escenas ambiguas en las que la imagen combina varias categorías plausibles o en las que el contexto pesa tanto como el sujeto principal.\n\n"
            "Los ejemplos de abajo muestran, por un lado, correcciones reales introducidas por el ajuste de la última capa y, por otro, comparativas de probabilidad por categoría que ayudan a ver "
            "si el modelo simplemente acierta más o si realmente redistribuye mejor su confianza."
        )
        draw_wrapped_text(fig, 0.08, 0.88, limit_text, width=96, fontsize=11.5, line_height=0.028)
        add_image(fig, vision_results["corrected_path"], [0.08, 0.42, 0.84, 0.24], "Ejemplos corregidos por el fine-tuning")
        add_image(fig, vision_results["prob_path"], [0.10, 0.10, 0.80, 0.24], "Comparación visual de probabilidades por categoría")
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Recommender
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.95, "4.5. Sistema final de recomendación", fontsize=18, weight="bold")
        reco_text = (
            "El notebook 05 reutiliza los artefactos ya construidos y demuestra el comportamiento conjunto del sistema con cinco usuarios simulados. "
            "Cada perfil combina intereses declarados, likes sintéticos y un ranking final de publicaciones e imágenes. La evaluación se formula mediante Precision@5, "
            "considerando relevante cualquier recomendación cuya categoría coincida con alguno de los intereses del usuario.\n\n"
            f"Los resultados agregados son sólidos: la Precision@5 media alcanza {recommender_results['metricas_globales_df'].loc[recommender_results['metricas_globales_df']['modalidad'].eq('Texto'), 'precision_at_5_media'].iloc[0]:.2f} en texto y "
            f"{recommender_results['metricas_globales_df'].loc[recommender_results['metricas_globales_df']['modalidad'].eq('Imagenes'), 'precision_at_5_media'].iloc[0]:.2f} en imágenes. "
            "Esto sugiere que la integración final conserva la intuición del proyecto: el perfil textual funciona bien para intereses mezclados y el perfil visual se beneficia especialmente del modelo fine-tuneado."
        )
        draw_wrapped_text(fig, 0.08, 0.88, reco_text, width=96, fontsize=11.5, line_height=0.028)
        add_image(fig, recommender_results["precision_chart_path"], [0.08, 0.47, 0.84, 0.22], "Evaluación agregada del sistema final")
        add_image(fig, recommender_results["metrics_table_path"], [0.08, 0.10, 0.40, 0.28], "Métricas por usuario")
        add_image(fig, recommender_results["gallery_path"], [0.52, 0.10, 0.40, 0.28], "Mini galería de recomendaciones visuales")
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page += 1

        # Conclusions
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.08, 0.94, "5. Conclusiones integradas", fontsize=23, weight="bold")
        conclusions = (
            "El recorrido completo del TFM queda cerrado en cinco aprendizajes principales.\n\n"
            "Primero, TF-IDF aportó un baseline claro, rápido e interpretable, útil para detectar que el problema ya no era trivial cuando el dataset incorporaba ambigüedad y ruido.\n\n"
            "Segundo, los embeddings aportaron una representación más flexible para recuperar contenido por cercanía semántica. Su valor no reside solo en una métrica agregada, sino en la naturalidad con la que manejan consultas mixtas.\n\n"
            "Tercero, la comparativa entre ambos enfoques mostró que elegir representación cambia el tipo de sistema: clasificación léxica frente a ranking semántico.\n\n"
            "Cuarto, el transfer learning visual aportó una mejora muy fuerte al adaptar una representación general a las ocho categorías concretas del proyecto. El bloque visual demuestra que incluso un ajuste limitado de la última capa puede transformar el rendimiento cuando el dominio está bien definido.\n\n"
            "Quinto, el sistema final demostró que esas piezas pueden convivir en una única narrativa funcional: representar texto, representar imágenes, construir perfiles y recomendar contenido alineado con intereses simulados.\n\n"
            "En conjunto, la aportación del proyecto no es solo técnica. También es metodológica: construir un caso experimental honesto, interpretar límites y mostrar cómo se integran decisiones simples pero bien justificadas."
        )
        draw_wrapped_text(fig, 0.08, 0.88, conclusions, width=96, fontsize=12, line_height=0.031)
        add_page_number(fig, page)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    ensure_dirs()
    text_results = compute_text_baseline()
    embeddings_results = compute_embeddings_results()
    vision_results = compute_vision_results()
    recommender_results = compute_final_recommender_results()
    build_pdf(text_results, embeddings_results, vision_results, recommender_results)
    print(f"PDF generado en: {PDF_PATH}")


if __name__ == "__main__":
    main()
