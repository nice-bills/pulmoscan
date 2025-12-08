
from fastapi import APIRouter
from app.schemas import HealthResponse
from app.services.model import classifier
from app.database import engine

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check system health.
    """
    # Simple check for now
    db_status = "healthy" 
    redis_status = "healthy" # Todo: Real check
    
    return {
        "status": "healthy",
        "services": {
            "postgres": db_status,
            "redis": redis_status,
            "model": "loaded" if classifier.model else "not_loaded"
        },
        "model_loaded": classifier.model is not None
    }
