"""Training event routes for recording trainer status via API."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.metrics import record_training_completion

router = APIRouter(prefix="/internal/training", tags=["Training"])

_ALLOWED_MODEL_TYPES = {"xgboost", "feedforward_nn", "multinomial_logistic"}
_ALLOWED_STATUSES = {"success", "failure"}


class TrainingCompletionEvent(BaseModel):
    model_type: str = Field(..., description="Model type identifier (e.g., xgboost)")
    status: str = Field(..., description="Training status: success or failure")
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional duration of the training run in seconds",
    )
    alias: Optional[str] = Field(
        default=None,
        description="Optional alias associated with the training run (Production/B/shadow1/shadow2)",
    )
    source: Optional[str] = Field(
        default="trainer",
        description="Origin of the training event for auditing purposes",
    )


@router.post("/complete")
def training_complete(event: TrainingCompletionEvent) -> dict:
    """Record a training completion event for Prometheus metrics."""
    model_type = event.model_type.strip().lower()
    status = event.status.strip().lower()

    if model_type not in _ALLOWED_MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {event.model_type}")
    if status not in _ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status: {event.status}")

    # Record the metric in Prometheus
    record_training_completion(model_type=model_type, status=status, duration_seconds=event.duration_seconds)

    return {
        "success": True,
        "model_type": model_type,
        "status": status,
        "duration_seconds": event.duration_seconds,
        "alias": event.alias,
        "source": event.source,
        "message": "Training completion event recorded",
    }
