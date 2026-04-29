from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    comentario: str
    categoria: str
    subcategoria: str
    sentimiento: str | None = None
    producto: str | None = None
    detalle: str | None = None


class PredictionResponse(BaseModel):
    predicted_classification: str
    prediction_confidence: float | None
    recommendation: str


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    year: int | None
    quarter: str | None
    date: date | None
    month: str | None
    source: str | None
    external_id: str | None
    comment: str | None
    sentiment: str | None
    category: str | None
    subcategory: str | None
    product: str | None
    detail: str | None
    original_classification: str | None
    predicted_classification: str | None
    prediction_confidence: float | None
    created_at: datetime
