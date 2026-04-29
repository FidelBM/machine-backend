from datetime import date, datetime
from io import BytesIO

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .ml import ModelCompatibilityError, get_model_warning, load_artifacts, predict, recommendation_for
from .models import Review
from .schemas import PredictRequest, PredictionResponse, ReviewOut


REQUIRED_COLUMNS = [
    "Año",
    "Trimestre",
    "Fecha",
    "Mes",
    "Fuente",
    "ID",
    "Comentario",
    "Sentimiento",
    "Categoria",
    "Subcategoria",
    "Producto",
    "Detalle",
    "Clasificacion",
]

app = FastAPI(title="ELEMENT ELITE FLEET API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    load_artifacts()


def parse_date(value: object) -> date | None:
    if pd.isna(value) or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def clean_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def review_from_row(row: pd.Series, allow_prediction_fallback: bool = False) -> tuple[Review, str | None]:
    payload = {
        "comment": clean_value(row.get("Comentario")),
        "category": clean_value(row.get("Categoria")),
        "subcategory": clean_value(row.get("Subcategoria")),
        "sentiment": clean_value(row.get("Sentimiento")),
        "product": clean_value(row.get("Producto")),
        "detail": clean_value(row.get("Detalle")),
    }
    prediction_warning = None
    try:
        predicted, confidence = predict(payload)
    except ModelCompatibilityError as exc:
        if not allow_prediction_fallback:
            raise
        predicted = clean_value(row.get("Clasificacion")) or "Pendiente de prediccion"
        confidence = None
        prediction_warning = str(exc)

    review = Review(
        year=int(row["Año"]) if not pd.isna(row.get("Año")) else None,
        quarter=clean_value(row.get("Trimestre")),
        date=parse_date(row.get("Fecha")),
        month=clean_value(row.get("Mes")),
        source=clean_value(row.get("Fuente")),
        external_id=clean_value(row.get("ID")),
        comment=payload["comment"],
        sentiment=payload["sentiment"],
        category=payload["category"],
        subcategory=payload["subcategory"],
        product=payload["product"],
        detail=payload["detail"],
        original_classification=clean_value(row.get("Clasificacion")),
        predicted_classification=predicted,
        prediction_confidence=confidence,
    )
    return review, prediction_warning


@app.post("/predict", response_model=PredictionResponse)
def predict_review(payload: PredictRequest) -> PredictionResponse:
    try:
        predicted, confidence = predict(payload.model_dump())
    except ModelCompatibilityError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(
        predicted_classification=predicted,
        prediction_confidence=confidence,
        recommendation=recommendation_for(predicted, confidence),
    )


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict[str, int | list[str]]:
    content = await file.read()
    try:
        dataframe = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_columns": missing})

    inserted = 0
    skipped = 0
    errors = 0
    error_details: list[str] = []
    prediction_warnings: set[str] = set()

    for _, row in dataframe.iterrows():
        external_id = clean_value(row.get("ID"))
        if external_id:
            exists = db.scalar(select(Review.id).where(Review.external_id == external_id))
            if exists:
                skipped += 1
                continue
        try:
            review, prediction_warning = review_from_row(row, allow_prediction_fallback=True)
            if prediction_warning:
                prediction_warnings.add(prediction_warning)
            db.add(review)
            db.commit()
            inserted += 1
        except IntegrityError:
            db.rollback()
            skipped += 1
        except Exception as exc:
            db.rollback()
            errors += 1
            if len(error_details) < 5:
                error_details.append(f"ID {external_id or 'sin ID'}: {exc}")

    return {
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
        "error_details": error_details,
        "prediction_warnings": sorted(prediction_warnings),
    }


@app.get("/reviews", response_model=list[ReviewOut])
def get_reviews(
    db: Session = Depends(get_db),
    category: str | None = None,
    subcategory: str | None = None,
    sentiment: str | None = None,
    product: str | None = None,
    classification: str | None = None,
    search: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[Review]:
    query = select(Review).order_by(Review.created_at.desc()).limit(limit).offset(offset)
    if category:
        query = query.where(Review.category == category)
    if subcategory:
        query = query.where(Review.subcategory == subcategory)
    if sentiment:
        query = query.where(Review.sentiment == sentiment)
    if product:
        query = query.where(Review.product == product)
    if classification:
        query = query.where(Review.predicted_classification == classification)
    if search:
        pattern = f"%{search}%"
        query = query.where(or_(Review.comment.ilike(pattern), Review.detail.ilike(pattern)))
    return list(db.scalars(query))


@app.get("/dashboard-summary")
def dashboard_summary(db: Session = Depends(get_db)) -> dict[str, object]:
    total = db.scalar(select(func.count(Review.id))) or 0
    churn_total = db.scalar(
        select(func.count(Review.id)).where(Review.predicted_classification.ilike("%abandono%"))
    ) or 0
    retention_total = db.scalar(
        select(func.count(Review.id)).where(Review.predicted_classification.ilike("%retenc%"))
    ) or 0
    if retention_total == 0 and total:
        retention_total = total - churn_total

    by_category = [
        {"name": name or "Sin categoria", "value": count}
        for name, count in db.execute(select(Review.category, func.count()).group_by(Review.category))
    ]
    by_subcategory = [
        {"name": name or "Sin subcategoria", "value": count}
        for name, count in db.execute(select(Review.subcategory, func.count()).group_by(Review.subcategory))
    ]
    trend = [
        {"name": name or "Sin mes", "value": count}
        for name, count in db.execute(select(Review.month, func.count()).group_by(Review.month).order_by(Review.month))
    ]

    return {
        "total_reviews": total,
        "high_churn_intent": churn_total,
        "likely_retention": retention_total,
        "by_category": by_category,
        "by_subcategory": by_subcategory,
        "trend": trend,
    }


@app.get("/categories")
def categories(db: Session = Depends(get_db)) -> dict[str, list[str]]:
    def distinct(column):
        rows = db.execute(select(column).where(column.is_not(None)).distinct().order_by(column)).scalars()
        return [value for value in rows if value]

    return {
        "categories": distinct(Review.category),
        "subcategories": distinct(Review.subcategory),
        "sentiments": distinct(Review.sentiment),
        "products": distinct(Review.product),
        "classifications": distinct(Review.predicted_classification),
    }


@app.get("/model-status")
def model_status() -> dict[str, str | bool | None]:
    warning = get_model_warning()
    return {"ready": warning is None, "warning": warning}
