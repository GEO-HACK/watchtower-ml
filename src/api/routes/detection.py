from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from api.schemas import DetectResponse, DetectionMeta, HealthResponse, PredictionResult
from api.services.detection_service import (
    DetectionInputError,
    DetectionNotReadyError,
    DetectionService,
    get_detection_service,
)


router = APIRouter(tags=["detection"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="watchtower-detection-api",
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(
    csv_file: Optional[UploadFile] = File(default=None),
    csv_path_form: Optional[str] = Form(default=None),
    csv_path: Optional[str] = Query(default=None),
    detection_service: DetectionService = Depends(get_detection_service),
) -> DetectResponse:
    try:
        selected_csv_path = csv_path or csv_path_form

        upload_filename = None
        upload_bytes = None
        if csv_file is not None:
            upload_filename = csv_file.filename
            upload_bytes = await csv_file.read()
            if not upload_bytes:
                raise DetectionInputError("Uploaded CSV file is empty")

        predictions, latency, flows_processed = detection_service.detect_from_csv(
            csv_path=selected_csv_path,
            upload_filename=upload_filename,
            upload_bytes=upload_bytes,
        )
        now = datetime.now(timezone.utc)

        single_prediction = PredictionResult(**predictions[0]) if len(predictions) == 1 else None
        batch_predictions = [PredictionResult(**item) for item in predictions] if len(predictions) > 1 else None

        return DetectResponse(
            status="success",
            processing_status="completed",
            timestamp=now,
            prediction=single_prediction,
            predictions=batch_predictions,
            meta=DetectionMeta(flows_processed=flows_processed, processing_latency_ms=latency),
        )
    except DetectionInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DetectionNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected detection error: {exc}",
        ) from exc
