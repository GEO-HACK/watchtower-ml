from fastapi import FastAPI

from api.config import settings
from api.routes.detection import router as detection_router


app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description="REST API for Watchtower hybrid ML network traffic detection",
)

app.include_router(detection_router)
