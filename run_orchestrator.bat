@echo off
setlocal enabledelayedexpansion
REM Universal MCP Orchestrator - Fixed Windows Run Script

echo ================================================================================
echo                        Universal MCP Orchestrator
echo                   AI Model and MCP Server Integration
echo ================================================================================
echo.

cd /d "%~dp0"

REM Check if enhanced setup has been run
if not exist "mcp_orchestrator.py" (
    echo ❌ Enhanced MCP Orchestrator files not found!
    echo.
    echo It looks like you need to set up the orchestrator first.
    echo Please follow these steps:
    echo.
    echo 1. Download all the enhanced files from the repository
    echo 2. Copy them to your Universal MCP Server directory
    echo 3. Run: python setup_mcp_orchestrator.py
    echo 4. Then run this script again
    echo.
    echo Alternatively, run simple_setup.bat first for a basic demo.
    echo.
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Check Node.js
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Node.js is not installed!
    echo.
    echo Node.js is required for MCP servers ^(web search, filesystem, git^).
    echo Please install Node.js from https://nodejs.org/
    echo.
    echo You can still use the orchestrator without MCP servers.
    echo Would you like to continue anyway? ^(Y/N^)
    set /p choice=
    if /I not "!choice!"=="Y" (
        exit /b 1
    )
) else (
    echo ✅ Node.js found: 
    node --version
)

echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  Virtual environment not found. Using system Python.
)

REM Check MCP dependencies
echo 🔍 Checking MCP dependencies...
python -c "import mcp" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  MCP dependencies not found. Installing...
    pip install mcp anthropic-mcp
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to install MCP dependencies
        echo Please run: pip install mcp anthropic-mcp
        pause
        exit /b 1
    )
)

REM Check configuration
if not exist ".env" (
    echo ⚠️  Configuration file .env not found!
    echo.
    if exist ".env.template" (
        echo Creating .env from template...
        copy ".env.template" ".env"
        echo.
        echo ✏️  Please edit .env and add your API keys:
        echo     • OpenAI API Key ^(optional^)
        echo     • Gemini API Key ^(optional^) 
        echo     • Anthropic API Key ^(optional^)
        echo.
        echo You need at least ONE API key configured.
        echo.
        start notepad.exe .env
        echo After saving your API keys, press any key to continue...
        pause >nul
    ) else (
        echo Please create a .env file with your API keys.
        pause
        exit /b 1
    )
)

REM Create logs directory
if not exist "logs" mkdir logs

REM Install MCP servers if Node.js is available
node --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🌐 Setting up MCP servers...
    
    REM Install filesystem server
    echo Installing filesystem MCP server...
    npm install -g @modelcontextprotocol/server-filesystem >nul 2>&1
    
    REM Install git server
    echo Installing git MCP server...
    npm install -g @modelcontextprotocol/server-git >nul 2>&1
    
    echo ✅ MCP servers installed
)

echo.
echo ================================================================================
echo                         Starting MCP Orchestrator
echo ================================================================================
echo.
echo 🚀 Universal MCP Orchestrator is starting...
echo.
echo ✨ Features available:
echo    • Multi-provider AI model selection ^(OpenAI, Gemini, Anthropic^)
echo    • MCP server integration ^(web search, filesystem, git^)
echo    • Interactive chat interface with tool visualization
echo    • Real-time web search without API keys
echo.
echo 🌐 Open your browser to: http://localhost:8080
echo.
echo 💡 Try asking: "What are the latest AI developments in 2025?"
echo    with web search enabled!
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the orchestrator
python run_orchestrator.py

REM Handle exit
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ The orchestrator stopped with an error.
    echo.
    echo Common issues:
    echo   • API keys not configured in .env
    echo   • Required dependencies not installed  
    echo   • Port 8080 already in use
    echo   • MCP server connection issues
    echo.
    echo Check logs\server.log for details.
) else (
    echo.
    echo 👋 Universal MCP Orchestrator stopped.
)

REM Deactivate virtual environment
if exist "venv\Scripts\activate.bat" (
    deactivate 2>nul
)

echo.
echo Press any key to exit...
pause >nul