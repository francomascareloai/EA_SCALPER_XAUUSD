from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analysis
from app.models import HealthResponse
import uvicorn
import time
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent_hub.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application Start Time
START_TIME = time.time()

app = FastAPI(
    title="EA_SCALPER_XAUUSD Agent Hub",
    description="Python Agent Hub for Advanced Analysis and Reasoning",
    version="2.0.0"
)

# CORS Middleware (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # ms
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

# Include Routers
app.include_router(analysis.router)

@app.get("/")
async def root():
    return {
        "message": "EA_SCALPER_XAUUSD Agent Hub is Running 🚀",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "technical": "/api/v1/technical",
            "fundamental": "/api/v1/fundamental",
            "sentiment": "/api/v1/sentiment"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="2.0.0",
        uptime_seconds=uptime
    )

if __name__ == "__main__":
    logger.info("Starting EA_SCALPER_XAUUSD Agent Hub...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
