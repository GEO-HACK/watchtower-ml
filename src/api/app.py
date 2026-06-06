# src/api/app.py
import httpx
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.config import settings
from api.routes.detection import router as detection_router

FLOW_API_URL = os.getenv("FLOW_API_URL", "http://localhost:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wait for signature module to be reachable before accepting traffic
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{FLOW_API_URL}/health")
            if r.status_code == 200:
                print(f"✓ Signature module reachable at {FLOW_API_URL}")
                break
        except Exception:
            print(f"  Waiting for signature module... ({attempt}/{max_retries})")
            await asyncio.sleep(3)
    else:
        print(f"⚠ Signature module not reachable after {max_retries} attempts — starting anyway")

    yield  # App runs here


app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description="REST API for Watchtower hybrid ML network traffic detection",
    lifespan=lifespan,
)

app.include_router(detection_router)