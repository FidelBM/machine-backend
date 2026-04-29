from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_PATHS = [Path("modelo_churn_2.pkl"), Path("modelo_reseñas_2.pkl")]
FEATURE_COLUMNS_PATH = Path("feature_columns_2.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model_ml: Any | None = None
model_emb: SentenceTransformer | None = None
feature_columns: list[str] = []
model_warning: str | None = None
embedding_dimension: int | None = None


class ModelCompatibilityError(RuntimeError):
    pass


def load_artifacts() -> None:
    global model_ml, model_emb, feature_columns, model_warning, embedding_dimension
    model_path = next((path for path in MODEL_PATHS if path.exists()), None)
    if model_path is None:
        names = ", ".join(str(path) for path in MODEL_PATHS)
        raise FileNotFoundError(f"No se encontro un modelo ML. Archivos esperados: {names}")
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError("No se encontro feature_columns_2.pkl")

    model_ml = joblib.load(model_path)
    feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))
    model_emb = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dimension = int(model_emb.get_sentence_embedding_dimension() or 0)
    expected_features = getattr(model_ml, "n_features_in_", None)
    generated_features = embedding_dimension + len(feature_columns)
    if expected_features is not None and expected_features != generated_features:
        model_warning = (
            f"El modelo {model_path.name} espera {expected_features} features, pero el backend genera "
            f"{generated_features}: {embedding_dimension} embeddings + {len(feature_columns)} columnas categoricas."
        )
    else:
        model_warning = None


def assert_model_compatible() -> None:
    if model_ml is None or model_emb is None:
        raise RuntimeError("Los modelos no han sido cargados.")
    if model_warning:
        raise ModelCompatibilityError(model_warning)


def get_model_warning() -> str | None:
    return model_warning


def _text(value: str | None) -> str:
    return (value or "").strip()


def build_categorical_frame(payload: dict[str, Any]) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        [
            {
                "Categoria": _text(payload.get("categoria") or payload.get("category")) or None,
                "Subcategoria": _text(payload.get("subcategoria") or payload.get("subcategory")) or None,
            }
        ]
    )
    dataframe = pd.get_dummies(dataframe)
    for column in feature_columns:
        if column not in dataframe:
            dataframe[column] = 0
    return dataframe[feature_columns]


def build_feature_matrix(payload: dict[str, Any]) -> np.ndarray:
    if model_emb is None:
        raise RuntimeError("El modelo de embeddings no ha sido cargado.")
    text = _text(payload.get("comentario") or payload.get("comment"))
    embedding = model_emb.encode([str(text)])
    categorical_values = build_categorical_frame(payload).values
    return np.hstack([embedding, categorical_values])


def predict(payload: dict[str, Any]) -> tuple[str, float | None]:
    assert_model_compatible()

    feature_matrix = build_feature_matrix(payload)
    prediction = model_ml.predict(feature_matrix)[0]
    confidence = None

    if hasattr(model_ml, "predict_proba"):
        probabilities = model_ml.predict_proba(feature_matrix)[0]
        classes = list(getattr(model_ml, "classes_", []))
        if 1 in classes:
            churn_probability = float(probabilities[classes.index(1)])
        elif "Alta intención de abandono" in classes:
            churn_probability = float(probabilities[classes.index("Alta intención de abandono")])
        else:
            churn_probability = float(max(probabilities))
        prediction = 1 if churn_probability > 0.5 else 0
        confidence = churn_probability if prediction == 1 else 1 - churn_probability

    classification = "Alta intención de abandono" if str(prediction) in {"1", "True"} else "Retención probable"
    return classification, confidence


def recommendation_for(classification: str, confidence: float | None) -> str:
    normalized = classification.lower()
    if "abandono" in normalized or "churn" in normalized or "alta" in normalized:
        return (
            "Priorizar contacto ejecutivo en menos de 24 horas, revisar causa raiz "
            "del comentario y proponer un plan de recuperacion con responsable comercial."
        )
    if confidence is not None and confidence < 0.6:
        return (
            "Validar manualmente el caso antes de automatizar acciones; la confianza "
            "del modelo es moderada y conviene enriquecer el contexto del cliente."
        )
    return (
        "Mantener seguimiento preventivo, reforzar los puntos positivos detectados "
        "y programar comunicacion de continuidad con el cliente."
    )
