from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool


class PredictResponse(BaseModel):
    label: str = Field(..., json_schema_extra={"example": "dog"})
    probabilities: dict
    confidence: float
    latency_ms: float
