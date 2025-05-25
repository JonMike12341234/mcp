@echo off
REM Universal MCP Server - Windows Installation and Run Script
REM This script will install dependencies and run the MCP server

echo ================================================================================
echo                          Universal MCP Server
echo                     Multi-Provider LLM Integration
echo ================================================================================
echo.

REM Set script directory as working directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment
        echo Make sure you have permissions to write to this directory.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
)

REM Check if venv\Scripts\activate.bat exists before activating
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv\Scripts\activate.bat not found!
    echo The virtual environment was not created correctly.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure you have permissions and that venv\Scripts\activate.bat exists.
    pause
    exit /b 1
)

REM Ensure pip is properly installed in the virtual environment
echo Ensuring pip is properly installed...
python -m ensurepip --upgrade
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: ensurepip failed, trying alternative method...
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py --force-reinstall
    del get-pip.py
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install pip
        echo Please ensure you have internet connectivity.
        pause
        exit /b 1
    )
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to upgrade pip
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Install or upgrade requirements
echo.
echo Installing/updating dependencies...
echo This may take a few minutes...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install requirements
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ================================================================================
    echo                            FIRST TIME SETUP
    echo ================================================================================
    echo.
    echo It looks like this is your first time running the Universal MCP Server.
    echo You need to configure your API keys before the server can start.
    echo.
    
    if exist ".env.template" (
        echo Copying environment template...
        copy ".env.template" ".env"
        echo.
        echo A .env file has been created from the template.
        echo Please edit the .env file and add your API keys:
        echo.
        echo   - OpenAI API Key: https://platform.openai.com/api-keys
        echo   - Google Gemini API Key: https://makersuite.google.com/app/apikey
        echo   - Anthropic API Key: https://console.anthropic.com/account/keys
        echo.
        echo You need at least ONE API key to use the server.
        echo.
        echo Opening .env file for editing...
        start notepad.exe .env
        echo.
        echo After saving your API keys, run this script again to start the server.
    ) else (
        echo .env.template not found. Please create a .env file manually with your API keys.
    )
    echo.
    pause
    exit /b 0
)

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "backups" mkdir backups

REM Check if at least one API key is configured
echo Checking configuration...
python -c "import os; from dotenv import load_dotenv; load_dotenv(); keys = [os.getenv('OPENAI_API_KEY'), os.getenv('GEMINI_API_KEY'), os.getenv('ANTHROPIC_API_KEY')]; configured = [p for p,k in zip(['OpenAI', 'Gemini', 'Anthropic'], keys) if k and k.strip() and k != f'your-{p.lower()}-api-key-here']; print('ERROR: No API keys configured!^nPlease edit the .env file and add at least one API key.') if not configured else print('Configured providers:', ', '.join(configured))"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Please edit the .env file and configure your API keys.
    echo Opening .env file...
    start notepad.exe .env
    echo.
    echo Run this script again after saving your configuration.
    pause
    exit /b 1
)

echo Configuration validated successfully!
echo.

REM Start the server
echo ================================================================================
echo                            Starting MCP Server
echo ================================================================================
echo.
echo The Universal MCP Server is starting...
echo.
echo You can:
echo   - Use the server with Claude Desktop (see README for setup)
echo   - Access the web UI at http://localhost:8080 (if enabled)
echo   - View logs in the logs/ directory
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the main server
python main.py

REM Check exit code
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================================================
    echo                              Server Stopped
    echo ================================================================================
    echo.
    echo The server stopped with an error. Check the logs for details.
    echo.
    echo Common issues:
    echo   - Invalid API keys
    echo   - Network connectivity problems  
    echo   - Port already in use
    echo   - Missing dependencies
    echo.
    echo Check logs/server.log for detailed error information.
    echo.
) else (
    echo.
    echo ================================================================================
    echo                         Server Stopped Cleanly
    echo ================================================================================
    echo.
    echo The Universal MCP Server has been stopped.
    echo Thank you for using our software!
    echo.
)

REM Deactivate virtual environment
deactivate

echo Press any key to exit...
pause >nul
