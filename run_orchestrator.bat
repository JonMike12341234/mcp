@echo off
echo ================================================================================
echo                        Universal MCP Orchestrator
echo ================================================================================
echo.

cd /d "%~dp0"

echo ✅ Starting setup...

REM Install Python dependencies quietly
if exist "venv\Scripts\python.exe" (
    echo Installing Python dependencies...
    venv\Scripts\python.exe -m pip install python-dotenv fastapi uvicorn pydantic >nul 2>&1
) else (
    pip install python-dotenv fastapi uvicorn pydantic >nul 2>&1
)

REM Check and install MCP servers if Node.js exists
node --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Checking MCP servers...
    
    where /q mcp-server-filesystem
    if %ERRORLEVEL% NEQ 0 (
        echo Installing filesystem server...
        npm install -g @modelcontextprotocol/server-filesystem >nul 2>&1
    ) else (
        echo ✅ Filesystem server already installed
    )
    
    where /q mcp-server-github
    if %ERRORLEVEL% NEQ 0 (
        echo Installing GitHub server...
        npm install -g @modelcontextprotocol/server-github >nul 2>&1
    ) else (
        echo ✅ GitHub server already installed
    )
    
    where /q mcp-server-memory
    if %ERRORLEVEL% NEQ 0 (
        echo Installing memory server...
        npm install -g @modelcontextprotocol/server-memory >nul 2>&1
    ) else (
        echo ✅ Memory server already installed
    )
    
    echo ✅ MCP servers check completed
)

echo.
echo 🚀 Starting Universal MCP Orchestrator...
echo 🌐 Open http://localhost:8080 in your browser
echo.

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe run_orchestrator.py
) else (
    python run_orchestrator.py
)

pause