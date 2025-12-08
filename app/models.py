
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ModelType(str, enum.Enum):
    COVID = "covid"
    SATELLITE = "satellite"

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    model_type = Column(Enum(ModelType), default=ModelType.COVID)
    
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    cached_images = Column(Integer, default=0)
    cache_hit_rate = Column(Float, default=0.0)
    
    celery_task_id = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    predictions = relationship("Prediction", back_populates="job")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    
    image_filename = Column(String)
    image_hash = Column(String, index=True)
    
    predicted_class = Column(String)
    confidence = Column(Float)
    top_3_classes = Column(JSON)
    
    processing_time_ms = Column(Float)
    from_cache = Column(Boolean, default=False)
    
    job = relationship("Job", back_populates="predictions")
