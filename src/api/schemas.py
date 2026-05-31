from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    """JSON payload for local CSV-path based detection."""

    csv_path: Optional[str] = Field(
        default=None,
        description="Local CSV path for testing (used when no file upload is provided).",
    )


class PredictionResult(BaseModel):
    label: str = Field(description="Normal or Attack")
    attack_type: Optional[str] = Field(default=None, description="Attack category if available")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DetectionMeta(BaseModel):
    flows_processed: int
    processing_latency_ms: Dict[str, float]


class DetectResponse(BaseModel):
    status: str
    processing_status: str
    timestamp: datetime
    prediction: Optional[PredictionResult] = None
    predictions: Optional[List[PredictionResult]] = None
    meta: DetectionMeta


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
