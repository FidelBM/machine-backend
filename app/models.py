from datetime import datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (UniqueConstraint("external_id", name="uq_reviews_external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quarter: Mapped[str | None] = mapped_column(String(50), nullable=True)
    date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    month: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    source: Mapped[str | None] = mapped_column(String(120), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    category: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    subcategory: Mapped[str | None] = mapped_column(String(220), nullable=True, index=True)
    product: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_classification: Mapped[str | None] = mapped_column(String(160), nullable=True)
    predicted_classification: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    prediction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
