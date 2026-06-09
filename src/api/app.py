# src/api/app.py
import httpx
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.config import settings
from api.routes.detection import router as detection_router
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    # Explicitly list your frontend URLs here. Add both localhost and 127.0.0.1 just to be safe!
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)