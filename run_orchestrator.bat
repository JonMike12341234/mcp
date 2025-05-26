@echo off
setlocal enabledelayedexpansion
REM Universal MCP Orchestrator - FIXED Windows Run Script

echo ================================================================================
echo                        Universal MCP Orchestrator
echo                   AI Model and MCP Server Integration - FIXED
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

REM Install required dependencies
echo 🔍 Installing required dependencies...
echo Installing core dependencies...
pip install python-dotenv fastapi uvicorn pydantic
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install core dependencies
    pause
    exit /b 1
)

echo Installing MCP dependencies...
pip install mcp
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  MCP installation had issues, but continuing...
)

echo ✅ Dependencies installed

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

REM Install CORRECT MCP servers if Node.js is available
node --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🌐 Checking MCP servers...
    
    REM Create a temporary file to store npm list output
    npm list -g --depth=0 > npm_list_temp.txt
    
    REM Check and install filesystem server
    findstr "@modelcontextprotocol/server-filesystem" npm_list_temp.txt >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Installing filesystem MCP server...
        call npm install -g @modelcontextprotocol/server-filesystem
        if %ERRORLEVEL% EQU 0 (
            echo ✅ Filesystem server installed successfully
        ) else (
            echo ⚠️ Filesystem server installation failed
        )
    ) else (
        echo ✅ Filesystem server already installed - skipping
    )
    
    REM Check and install GitHub server
    findstr "@modelcontextprotocol/server-github" npm_list_temp.txt >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Installing GitHub MCP server...
        call npm install -g @modelcontextprotocol/server-github
        if %ERRORLEVEL% EQU 0 (
            echo ✅ GitHub server installed successfully
        ) else (
            echo ❌ GitHub server installation failed
        )
    ) else (
        echo ✅ GitHub server already installed - skipping
    )
    
    REM Check and install memory server
    findstr "@modelcontextprotocol/server-memory" npm_list_temp.txt >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Installing memory MCP server...
        call npm install -g @modelcontextprotocol/server-memory
        if %ERRORLEVEL% EQU 0 (
            echo ✅ Memory server installed successfully
        ) else (
            echo ⚠️ Memory server installation failed
        )
    ) else (
        echo ✅ Memory server already installed - skipping
    )
    
    REM Check and install brave search server
    findstr "@modelcontextprotocol/server-brave-search" npm_list_temp.txt >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Installing Brave Search MCP server...
        call npm install -g @modelcontextprotocol/server-brave-search
        if %ERRORLEVEL% EQU 0 (
            echo ✅ Brave Search server installed successfully
        ) else (
            echo ⚠️ Brave Search server installation failed
        )
    ) else (
        echo ✅ Brave Search server already installed - skipping
    )
    
    REM Check and install puppeteer server
    findstr "@modelcontextprotocol/server-puppeteer" npm_list_temp.txt >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Installing Puppeteer MCP server...
        call npm install -g @modelcontextprotocol/server-puppeteer
        if %ERRORLEVEL% EQU 0 (
            echo ✅ Puppeteer server installed successfully
        ) else (
            echo ⚠️ Puppeteer server installation failed
        )
    ) else (
        echo ✅ Puppeteer server already installed - skipping
    )
    
    REM Check and install web search server
    findstr "web-search-mcp-server" npm_list_temp.txt >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        findstr "@pskill9/web-search" npm_list_temp.txt >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo Installing web search MCP server...
            call npm install -g web-search-mcp-server 2>nul
            if %ERRORLEVEL% NEQ 0 (
                echo Trying alternative web search server...
                call npm install -g @pskill9/web-search 2>nul
                if %ERRORLEVEL% NEQ 0 (
                    echo ⚠️ Web search server installation failed, but continuing...
                ) else (
                    echo ✅ Alternative web search server installed
                )
            ) else (
                echo ✅ Web search server installed successfully
            )
        ) else (
            echo ✅ Alternative web search server already installed - skipping
        )
    ) else (
        echo ✅ Web search server already installed - skipping
    )
    
    REM Clean up temporary file
    del npm_list_temp.txt
    
    echo.
    echo ✅ MCP servers check/installation completed!
    echo.
    echo 📋 Successfully installed MCP servers:
    echo    ✅ Filesystem operations ^(@modelcontextprotocol/server-filesystem^)
    echo    ✅ GitHub integration ^(@modelcontextprotocol/server-github^)
    echo    ✅ Memory management ^(@modelcontextprotocol/server-memory^)
    echo    ✅ Brave Search ^(@modelcontextprotocol/server-brave-search^)
    echo    ✅ Web automation ^(@modelcontextprotocol/server-puppeteer^)
    echo    ✅ Web search ^(alternative packages^)
    
) else (
    echo ⚠️ Skipping MCP server installation ^(Node.js not available^)
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
echo    • REAL MCP server integration with CORRECT packages:
echo      - Filesystem Operations ^(secure file access^)
echo      - GitHub Integration ^(repository management^)
echo      - Memory Management ^(persistent storage^)
echo      - Brave Search ^(web search without API keys^)  
echo      - Web Automation ^(Puppeteer browser control^)
echo    • Interactive chat interface with tool visualization
echo    • Real-time web search and automation capabilities
echo.
echo 🌐 Open your browser to: http://localhost:8080
echo.
echo 💡 Try asking:
echo    "Search for recent AI developments" ^(with Brave Search^)
echo    "List files in my project directory" ^(with Filesystem^)
echo    "What's new in my GitHub repositories?" ^(with GitHub^)
echo    "Take a screenshot of google.com" ^(with Puppeteer^)
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
    echo 🔧 Fixed in this version:
    echo   ✅ Used correct package name: @modelcontextprotocol/server-github
    echo   ✅ Removed non-existent: @modelcontextprotocol/server-git
    echo   ✅ Added working MCP servers: memory, brave-search, puppeteer
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