
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "PulmoScan"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str
    
    # Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "pulmoscan-images"
    
    # ML
    MODEL_PATH: str = "models/covid/mobilenetv3_best.pth"
    DEVICE: str = "cpu"
    BATCH_SIZE: int = 32
    CACHE_ENABLED: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
