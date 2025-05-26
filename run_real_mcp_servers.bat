@echo off
setlocal enabledelayedexpansion
REM Universal MCP Orchestrator - Updated for Real MCP Servers

echo ================================================================================
echo                        Universal MCP Orchestrator
echo                   Real MCP Server Integration (No API Keys!)
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
    echo Node.js is required for REAL MCP servers ^(Google search, Perplexity, file ops, etc.^).
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

REM Check Git
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Git is not installed!
    echo Git is needed to install real MCP servers from GitHub.
    echo Please install Git from https://git-scm.com/
    echo.
    echo You can still use the orchestrator without MCP servers.
    echo Would you like to continue anyway? ^(Y/N^)
    set /p choice=
    if /I not "!choice!"=="Y" (
        exit /b 1
    )
) else (
    echo ✅ Git found: 
    git --version
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

REM Setup Real MCP Servers
echo.
echo 🌐 Setting up REAL MCP servers from GitHub...
echo.

REM Check if real MCP setup script exists
if exist "real_mcp_setup_script.py" (
    echo Running real MCP servers setup...
    python real_mcp_setup_script.py
    if %ERRORLEVEL% NEQ 0 (
        echo ⚠️  Real MCP setup had issues, but continuing...
    )
) else (
    echo ⚠️  Real MCP setup script not found, installing manually...
    
    REM Install basic MCP servers if Node.js and Git are available
    node --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        git --version >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            echo Installing official MCP servers...
            
            REM Install official servers via npm
            echo Installing filesystem MCP server...
            call npm install -g @modelcontextprotocol/server-filesystem
            
            echo Installing memory MCP server...
            call npm install -g @modelcontextprotocol/server-memory
            
            REM Note: Git functionality is available through GitHub server (requires token)
            REM echo Installing GitHub MCP server (requires GITHUB_TOKEN)...
            REM call npm install -g @modelcontextprotocol/server-github
            
            REM Create mcp_servers directory
            if not exist "mcp_servers" mkdir mcp_servers
            
            echo Installing Google Web Search MCP server...
            if not exist "mcp_servers\web-search-google" (
                git clone https://github.com/pskill9/web-search.git mcp_servers\web-search-google
                if %ERRORLEVEL% EQU 0 (
                    cd mcp_servers\web-search-google
                    call npm install
                    call npm run build
                    cd ..\..
                    echo ✅ Google Web Search MCP server installed
                ) else (
                    echo ❌ Failed to install Google Web Search server
                )
            ) else (
                echo ✅ Google Web Search server already exists
            )
            
            echo Installing DuckDuckGo Web Search MCP server...
            if not exist "mcp_servers\web-search-duckduckgo" (
                git clone https://github.com/kouui/web-search-duckduckgo.git mcp_servers\web-search-duckduckgo
                if %ERRORLEVEL% EQU 0 (
                    echo ✅ DuckDuckGo Web Search MCP server installed
                ) else (
                    echo ❌ Failed to install DuckDuckGo Web Search server
                )
            ) else (
                echo ✅ DuckDuckGo Web Search server already exists
            )
            
            echo Installing Perplexity Web Search MCP server...
            if not exist "mcp_servers\web-search-perplexity" (
                git clone https://github.com/wysh3/perplexity-mcp-zerver.git mcp_servers\web-search-perplexity
                if %ERRORLEVEL% EQU 0 (
                    cd mcp_servers\web-search-perplexity
                    call npm install
                    call npm run build
                    cd ..\..
                    echo ✅ Perplexity Web Search MCP server installed
                ) else (
                    echo ❌ Failed to install Perplexity Web Search server
                )
            ) else (
                echo ✅ Perplexity Web Search server already exists
            )
            
            echo ✅ Real MCP servers setup complete
        ) else (
            echo ⚠️ Git not available, skipping GitHub-based MCP servers
        )
    ) else (
        echo ⚠️ Node.js not available, skipping MCP server installation
    )
)

echo.
echo ================================================================================
echo                         Starting Real MCP Orchestrator
echo ================================================================================
echo.
echo 🚀 Universal MCP Orchestrator is starting...
echo.
echo ✨ REAL Features available:
echo    • Multi-provider AI model selection ^(OpenAI, Gemini, Anthropic^)
echo    • REAL MCP server integration from GitHub:
echo      - Google Web Search ^(pskill9/web-search^)
echo      - DuckDuckGo Web Search ^(kouui/web-search-duckduckgo^) 
echo      - Perplexity AI Search ^(wysh3/perplexity-mcp-zerver^)
echo      - File System Operations ^(official MCP^)
echo      - Memory Operations ^(official MCP^)
echo    • Interactive chat interface with REAL tool visualization
echo    • Real-time web search WITHOUT API keys!
echo.
echo 🌐 Open your browser to: http://localhost:8080
echo.
echo 💡 Try asking: "What are the latest AI developments in 2025?"
echo    with Google Web Search or Perplexity selected!
echo.
echo 🔍 All search servers work WITHOUT API keys:
echo    • Google search scrapes results directly
echo    • DuckDuckGo fetches and converts content to markdown
echo    • Perplexity uses browser automation for AI-powered search
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
    echo   • Real MCP servers not built properly
    echo.
    echo Check logs\server.log for details.
    echo.
    echo 🔧 Troubleshooting:
    echo   • Make sure Node.js and Git are installed
    echo   • Check that mcp_servers directory exists
    echo   • Verify MCP servers were built successfully
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