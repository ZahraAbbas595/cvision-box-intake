from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    service_version: str
    model_name: str
    model_version: str
    model_backend: str
    schema_version: str
