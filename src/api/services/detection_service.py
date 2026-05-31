import logging
from threading import Lock
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from api.config import settings
from api.services.ml_inference_service import MLInferenceService
from ingestion.csv_ingestion import load_flows_from_csv_path, load_flows_from_csv_upload


logger = logging.getLogger(__name__)


class DetectionInputError(ValueError):
    pass


class DetectionNotReadyError(RuntimeError):
    pass


class DetectionService:
    """Detection service layer that orchestrates ingestion and ML inference."""

    def __init__(self) -> None:
        self.ml_inference = None
        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        try:
            self.ml_inference = MLInferenceService()
            logger.info("Detection runtime initialized")
        except Exception as exc:
            logger.exception("Failed to initialize detection runtime")
            raise DetectionNotReadyError(f"Detection runtime initialization failed: {exc}") from exc

    def detect_from_csv(
        self,
        csv_path: Optional[str] = None,
        upload_filename: Optional[str] = None,
        upload_bytes: Optional[bytes] = None,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float], int]:
        if upload_bytes is not None:
            if not upload_filename:
                upload_filename = "uploaded.csv"
            df, _ = load_flows_from_csv_upload(upload_filename, upload_bytes)
        else:
            resolved_csv_path = csv_path or settings.default_test_csv_path
            if not Path(resolved_csv_path).exists():
                raise DetectionInputError(f"CSV file not found: {resolved_csv_path}")
            df, _ = load_flows_from_csv_path(resolved_csv_path)

        if df.empty:
            raise DetectionInputError("No flows extracted from CSV")

        try:
            predictions, latency = self.ml_inference.predict_from_dataframe(df)
        except Exception as exc:
            raise DetectionInputError(f"Detection failed: {exc}") from exc

        return predictions, latency, len(df)


_detection_service = None
_detection_service_lock = Lock()


def get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        with _detection_service_lock:
            if _detection_service is None:
                _detection_service = DetectionService()
    return _detection_service
