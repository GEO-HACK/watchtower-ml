from datetime import datetime, timezone
from typing import Optional
import httpx
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from api.schemas import DetectResponse, DetectionMeta, HealthResponse, PredictionResult
from api.services.detection_service import (
    DetectionInputError,
    DetectionNotReadyError,
    DetectionService,
    get_detection_service,
)

router = APIRouter(tags=["detection"])

FLOW_API_URL = os.getenv("FLOW_API_URL", "http://localhost:8001/flows")


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="watchtower-detection-api",
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/detect/live", response_model=DetectResponse)
async def detect_live(
    detection_service: DetectionService = Depends(get_detection_service),
) -> DetectResponse:
    """
    Pulls live flows from signature module (localhost:8001/flows)
    and runs ML inference on them.

    Test from terminal:
        curl http://localhost:8002/detect/live | jq
    """
    # --- Fetch CSV flows from signature module ---
    try:
        async with httpx.AsyncClient(timeout=3000.0) as client:
            response = await client.get(FLOW_API_URL)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot reach signature module at {FLOW_API_URL}. Is it running on port 8001?",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Signature module timed out at {FLOW_API_URL}",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Signature module returned {response.status_code}: {response.text[:200]}",
        )

    csv_bytes = response.content

    if not csv_bytes:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Signature module returned empty flow data",
        )

    # --- Run inference ---
    try:
        predictions, latency, flows_processed = detection_service.detect_from_csv(
            csv_path=None,
            upload_filename="live_flows.csv",
            upload_bytes=csv_bytes,
        )
    except DetectionInputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DetectionNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    now = datetime.now(timezone.utc)

    single_prediction = PredictionResult(**predictions[0]) if len(predictions) == 1 else None
    batch_predictions = [PredictionResult(**p) for p in predictions] if len(predictions) > 1 else None

    return DetectResponse(
        status="success",
        processing_status="completed",
        timestamp=now,
        prediction=single_prediction,
        predictions=batch_predictions,
        meta=DetectionMeta(
            flows_processed=flows_processed,
            processing_latency_ms=latency,
        ),
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(
    csv_file: Optional[UploadFile] = File(default=None),
    csv_path_form: Optional[str] = Form(default=None),
    csv_path: Optional[str] = Query(default=None),
    detection_service: DetectionService = Depends(get_detection_service),
) -> DetectResponse:
    """
    Run inference on an uploaded CSV file or a path on disk.

    Test from terminal:
        curl -X POST http://localhost:8002/detect \
             -F "csv_file=@flows.csv" | jq
    """
    try:
        upload_filename = None
        upload_bytes = None
        selected_csv_path = csv_path or csv_path_form

        if csv_file is not None:
            upload_filename = csv_file.filename
            upload_bytes = await csv_file.read()
            if not upload_bytes:
                raise DetectionInputError("Uploaded CSV file is empty")

        if not upload_bytes and not selected_csv_path:
            raise DetectionInputError(
                "Provide a CSV file, a csv_path, or use GET /detect/live for live flows"
            )

        predictions, latency, flows_processed = detection_service.detect_from_csv(
            csv_path=selected_csv_path,
            upload_filename=upload_filename,
            upload_bytes=upload_bytes,
        )

        now = datetime.now(timezone.utc)

        single_prediction = PredictionResult(**predictions[0]) if len(predictions) == 1 else None
        batch_predictions = [PredictionResult(**p) for p in predictions] if len(predictions) > 1 else None

        return DetectResponse(
            status="success",
            processing_status="completed",
            timestamp=now,
            prediction=single_prediction,
            predictions=batch_predictions,
            meta=DetectionMeta(
                flows_processed=flows_processed,
                processing_latency_ms=latency,
            ),
        )

    except DetectionInputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DetectionNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected detection error: {exc}",
        )