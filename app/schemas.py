from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str


class VersionResponse(BaseModel):
    """Version metadata for the running service and inference model."""

    service_version: str
    model_name: str
    model_version: str
    model_backend: str
    schema_version: str
