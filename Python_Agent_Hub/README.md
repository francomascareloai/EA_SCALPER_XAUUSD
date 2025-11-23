# EA_SCALPER_XAUUSD Python Agent Hub

## Overview
Python-based analysis hub for the EA_SCALPER_XAUUSD trading system. Provides advanced technical, fundamental, and sentiment analysis via REST API.

## Architecture
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and schema enforcement
- **Modular Agents**: Separate services for each analysis type

## Project Structure
```
Python_Agent_Hub/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── app/
│   ├── models/
│   │   └── schemas.py      # Pydantic models (Request/Response)
│   ├── routers/
│   │   └── analysis.py     # API endpoints
│   └── services/
│       └── technical_agent.py  # Technical analysis logic
└── venv/                   # Virtual environment
```

## Installation

### 1. Create Virtual Environment
```bash
cd Python_Agent_Hub
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Edit `.env` file with your API keys:
```
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## Running the Server

### Development Mode (Auto-reload)
```bash
source venv/bin/activate
python main.py
```

### Production Mode
```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check
```
GET /health
```

### Technical Analysis (Fast Lane)
```
POST /api/v1/technical
Content-Type: application/json

{
  "schema_version": "1.0",
  "req_id": "unique_id",
  "timestamp": 1700000000,
  "timeout_ms": 200,
  "symbol": "XAUUSD",
  "timeframe": "M15",
  "current_price": 2050.50
}
```

**Response:**
```json
{
  "schema_version": "1.0",
  "req_id": "unique_id",
  "timestamp": 1700000000,
  "tech_subscore": 75,
  "signal_type": "long",
  "confidence": 0.75,
  "processing_time_ms": 150.5,
  "error": null
}
```

### Fundamental Analysis (Slow Lane)
```
POST /api/v1/fundamental
```

### Sentiment Analysis (Slow Lane)
```
POST /api/v1/sentiment
```

## MQL5 Integration

The MQL5 EA communicates with this hub via `PythonBridge.mqh`:

1. **Add URL to MT5 Whitelist**:
   - Tools → Options → Expert Advisors
   - Check "Allow WebRequest for listed URL"
   - Add: `http://127.0.0.1:8000`

2. **Initialize Bridge in EA**:
```cpp
#include <EA_Elite_Components/PythonBridge.mqh>

CPythonBridge g_PythonBridge;

int OnInit() {
    if(!g_PythonBridge.Init("http://127.0.0.1:8000")) {
        Print("Failed to connect to Python Hub");
        return INIT_FAILED;
    }
    return INIT_SUCCEEDED;
}
```

## Development Roadmap

### Phase 2 (Current - MVP)
- [x] FastAPI skeleton
- [x] Pydantic models
- [x] Technical Agent stub
- [x] MQL5 Bridge
- [ ] Integration testing

### Phase 3 (Advanced Features)
- [ ] Real technical indicators (TA-Lib)
- [ ] News API integration
- [ ] Sentiment data providers
- [ ] LLM reasoning (OpenAI/Gemini)
- [ ] Caching layer (Redis)
- [ ] Rate limiting
- [ ] Authentication

## Logging
Logs are written to:
- Console (stdout)
- `agent_hub.log` file

## Performance Targets
- **Technical Analysis**: < 200ms
- **Fundamental Analysis**: < 5s
- **Sentiment Analysis**: < 3s
- **LLM Reasoning**: < 10s

## Security Notes
- **Development**: CORS is open (`allow_origins=["*"]`)
- **Production**: Restrict CORS to specific origins
- **API Keys**: Never commit `.env` to version control
- **Authentication**: Add API key validation for production

## Troubleshooting

### Server won't start
- Check if port 8000 is available: `lsof -i :8000`
- Verify virtual environment is activated
- Check Python version: `python --version` (requires 3.10+)

### MQL5 can't connect
- Ensure server is running: `curl http://127.0.0.1:8000/health`
- Verify URL is whitelisted in MT5
- Check firewall settings

## License
Proprietary - EA_SCALPER_XAUUSD Project
