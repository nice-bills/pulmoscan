
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class PredictionBase(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    inference_time: float

class PredictionCreate(PredictionBase):
    pass

class PredictionResponse(PredictionBase):
    top_3_classes: Optional[List[Dict[str, float]]] = None
    from_cache: bool = False

class JobBase(BaseModel):
    total_images: int

class JobCreate(JobBase):
    pass

class JobResponse(JobBase):
    id: str
    status: JobStatus
    created_at: datetime
    processed_images: int
    failed_images: int
    cache_hit_rate: float
    
    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    model_loaded: bool
