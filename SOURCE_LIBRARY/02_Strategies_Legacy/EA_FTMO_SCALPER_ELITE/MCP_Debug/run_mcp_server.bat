@echo off
echo Starting MetaTrader 5 MCP Server...

:: Add the src directory to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%~dp0src

:: Install required packages if they're not already installed
pip install -q mcp-metatrader5-server uvicorn MetaTrader5

:: Run the server
python -m uvicorn mcp_metatrader5_server.server:mcp.app --host 127.0.0.1 --port 8000 --reload
