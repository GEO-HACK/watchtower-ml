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
    # Core
    label: str                        # 'Normal' | 'Attack'
    attack_type: Optional[str]        # 'DDoS', 'PortScan', etc. or None
    confidence: float

    # Per-model
    model1_label: Optional[str] = None
    model1_confidence: Optional[float] = None
    model2_label: Optional[str] = None
    model2_confidence: Optional[float] = None

    # Isolation Forest
    if_prediction: Optional[int] = None      # 1 = normal, -1 = anomaly
    if_anomaly_score: Optional[float] = None

    # Fusion
    fused_label: Optional[str] = None
    fused_score: Optional[float] = None
    escalated: Optional[bool] = None

    # Agreement
    models_agree: Optional[bool] = None

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
