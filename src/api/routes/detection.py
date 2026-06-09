from datetime import datetime, timezone
from typing import Optional
import httpx
import os

from joblib import logger

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from api.schemas import DetectResponse, DetectionMeta, HealthResponse, PredictionResult
from api.services.detection_service import (
    DetectionInputError,
    DetectionNotReadyError,
    DetectionService,
    get_detection_service,
)

router = APIRouter(tags=["detection"])

FLOW_API_URL = os.getenv("FLOW_API_URL", "http://localhost:8000/flows")
SIGNATURE_FUSION_URL = os.getenv("SIGNATURE_FUSION_URL", "http://localhost:8000/flows/ml-results")


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
    Pulls live flows from signature module (localhost:8000/flows)
    and runs ML inference on them.
    """
    # --- Fetch CSV flows from signature module ---
    try:
        async with httpx.AsyncClient(timeout=3000.0) as client:
            response = await client.get(FLOW_API_URL)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot reach signature module at {FLOW_API_URL}. Is it running on port 8000?",
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

    # --- Push to Signature Fusion Layer ---
    if batch_predictions:
        predictions_payload = [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in batch_predictions]
    elif single_prediction:
        predictions_payload = [single_prediction.model_dump() if hasattr(single_prediction, 'model_dump') else single_prediction.dict()]
    else:
        predictions_payload = []

    if predictions_payload:
        try:
            print(f"ML DEBUG: Pushing {len(predictions_payload)} predictions to {SIGNATURE_FUSION_URL}...")
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(SIGNATURE_FUSION_URL, json=predictions_payload, timeout=30.0)
                print(f"ML DEBUG: Fusion endpoint replied with status: {resp.status_code}")
                    
                    # --- NEW: Print the exact reason for the 422 error ---
                if resp.status_code != 200:
                    print(f"ML DEBUG ERROR DETAILS: {resp.text}")
                    # -----------------------------------------------------
                        
        except Exception as exc:
            print(f"ERROR: Failed to stream inference payload to fusion backend: {exc}")
            print(f"ML DEBUG ERROR: {exc}")
    # ---------------------------------------

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

        # --- Push to Signature Fusion Layer ---
        if batch_predictions:
            predictions_payload = [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in batch_predictions]
        elif single_prediction:
            predictions_payload = [single_prediction.model_dump() if hasattr(single_prediction, 'model_dump') else single_prediction.dict()]
        else:
            predictions_payload = []

        if predictions_payload:
            try:
                print(f"ML DEBUG: Pushing {len(predictions_payload)} predictions to {SIGNATURE_FUSION_URL}...")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(SIGNATURE_FUSION_URL, json=predictions_payload)
                    print(f"ML DEBUG: Fusion endpoint replied with status: {resp.status_code}")
            except Exception as exc:
                logger.error(f"Failed to stream inference payload to fusion backend: {exc}")
                print(f"ML DEBUG ERROR: {exc}")
        # ---------------------------------------

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