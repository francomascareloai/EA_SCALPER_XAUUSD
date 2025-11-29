"""
EA_SCALPER_XAUUSD Agent Hub v3.0
================================
Python Backend for Trading Analysis & Fundamentals

Endpoints:
- /health                    - Health check
- /api/v1/technical          - Technical analysis
- /api/v1/fundamentals       - Macro fundamentals (FRED)
- /api/v1/sentiment          - News sentiment (FinBERT)
- /api/v1/signal             - Aggregated trading signal
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("agent_hub.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application Start Time
START_TIME = time.time()

# Create FastAPI app
app = FastAPI(
    title="EA_SCALPER_XAUUSD Agent Hub",
    description="Python Backend for Gold Trading - Fundamentals, Sentiment & ML",
    version="3.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing Middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

# Import and include routers
from app.routers import fundamentals

app.include_router(fundamentals.router, prefix="/api/v1", tags=["fundamentals"])

@app.get("/")
async def root():
    return {
        "message": "EA_SCALPER_XAUUSD Agent Hub v3.0 - Fundamentals Edition",
        "version": "3.0.0",
        "endpoints": {
            "health": "/health",
            "fundamentals": "/api/v1/fundamentals",
            "sentiment": "/api/v1/sentiment", 
            "signal": "/api/v1/signal",
            "oil": "/api/v1/oil",
            "macro": "/api/v1/macro"
        }
    }

@app.get("/health")
async def health_check():
    uptime = time.time() - START_TIME
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "uptime_seconds": round(uptime, 2),
        "fred_configured": bool(os.getenv('FRED_API_KEY')),
        "newsapi_configured": bool(os.getenv('NEWSAPI_KEY'))
    }

if __name__ == "__main__":
    logger.info("Starting EA_SCALPER_XAUUSD Agent Hub v3.0...")
    logger.info(f"FRED API: {'Configured' if os.getenv('FRED_API_KEY') else 'NOT SET'}")
    logger.info(f"NewsAPI: {'Configured' if os.getenv('NEWSAPI_KEY') else 'NOT SET'}")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
