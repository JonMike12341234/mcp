@echo off
REM Universal MCP Server with Web Dashboard
REM Runs both the MCP server and web dashboard together

echo ================================================================================
echo                   Universal MCP Server + Web Dashboard
echo ================================================================================
echo.

REM Set script directory as working directory
cd /d "%~dp0"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found. Please run run.bat first.
    pause
    exit /b 1
)

echo Starting Universal MCP Server with Web Dashboard...
echo.
echo - MCP server will start in this window (for Claude Desktop)
echo - Web dashboard will open automatically in your browser
echo - Both services will run together
echo.
echo Press Ctrl+C to stop both services
echo.

REM Start web server in background and open browser
echo Starting web dashboard...
start /B python web_server.py

REM Give web server time to start
timeout /t 2 /nobreak >nul

REM Open browser
start http://localhost:8080

REM Start MCP server (this will block)
echo.
echo Starting MCP Server...
echo ================================================================================
python main.py

REM Cleanup message when MCP server stops
echo.
echo ================================================================================
echo MCP Server stopped. 
echo.
echo Note: The web dashboard may still be running in the background.
echo If you need to stop it manually, you can:
echo 1. Check Task Manager for python.exe processes
echo 2. Or restart your computer
echo 3. Or use: taskkill /f /im python.exe
echo ================================================================================
pause
