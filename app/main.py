from fastapi import FastAPI
from app.schemas import HealthResponse, VersionResponse

app = FastAPI(title="CVision Box Intake API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        service_version="0.1.0",
        model_name="yolov8s-worldv2",
        model_version="8.4.0",
        schema_version="1.0",
    )
