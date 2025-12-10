from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import health, jobs
from app.database import Base, engine
from app.config import settings
from app.workers.celery_app import celery_app # Ensure Celery app is loaded

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="High-performance COVID-19 X-ray Classification API",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(health.router, tags=["Health"])
app.include_router(jobs.router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Welcome to PulmoScan API. Visit /docs for documentation."}