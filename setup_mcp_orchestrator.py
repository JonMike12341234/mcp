#!/usr/bin/env python3
"""
Setup script for Universal MCP Orchestrator
Installs dependencies and sets up the environment for MCP orchestration
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False

def check_node_npm():
    """Check if Node.js and npm are installed."""
    print("🔍 Checking Node.js and npm installation...")
    
    node_check = subprocess.run("node --version", shell=True, capture_output=True)
    npm_check = subprocess.run("npm --version", shell=True, capture_output=True)
    
    if node_check.returncode != 0:
        print("❌ Node.js is not installed. Please install Node.js from https://nodejs.org/")
        return False
    
    if npm_check.returncode != 0:
        print("❌ npm is not installed. Please install npm (usually comes with Node.js)")
        return False
    
    print(f"   ✅ Node.js version: {node_check.stdout.decode().strip()}")
    print(f"   ✅ npm version: {npm_check.stdout.decode().strip()}")
    return True

def install_python_dependencies():
    """Install Python dependencies for MCP client support."""
    print("📦 Installing Python dependencies...")
    
    # Add MCP client dependencies
    mcp_requirements = [
        "mcp>=1.0.0",
        "anthropic-mcp>=0.1.0",  # MCP client for Python
    ]
    
    for req in mcp_requirements:
        if not run_command(f"pip install {req}", f"Installing {req}"):
            print(f"⚠️  Warning: Failed to install {req}")
    
    return True

def setup_mcp_servers():
    """Set up MCP servers directory and install basic servers."""
    print("🌐 Setting up MCP servers...")
    
    # Create MCP servers directory
    mcp_dir = Path("mcp_servers")
    mcp_dir.mkdir(exist_ok=True)
    
    # Install basic MCP servers via npm
    mcp_servers = [
        "@modelcontextprotocol/server-filesystem",
        "@modelcontextprotocol/server-git",
    ]
    
    for server in mcp_servers:
        run_command(
            f"npm install -g {server}", 
            f"Installing {server} globally",
            check=False
        )
    
    return True

def create_enhanced_requirements():
    """Create an enhanced requirements.txt with MCP support."""
    requirements_content = """# Universal MCP Orchestrator Requirements
# Core MCP and server dependencies
mcp>=1.0.0
asyncio-mqtt>=0.16.1
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
websockets>=12.0
pydantic>=2.5.0
python-multipart>=0.0.6

# MCP Client Support
anthropic-mcp>=0.1.0

# LLM Provider SDKs
openai>=1.51.0
google-generativeai>=0.8.3
anthropic>=0.37.1

# HTTP and API clients
httpx>=0.27.0
aiohttp>=3.9.1
requests>=2.31.0

# Configuration and environment
python-dotenv>=1.0.0
pyyaml>=6.0.1
toml>=0.10.2

# Security and encryption
cryptography>=41.0.8
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0

# Logging and monitoring
structlog>=23.2.0
prometheus-client>=0.19.0
psutil>=5.9.6

# Database (optional, for caching and session management)
redis>=5.0.1
aiosqlite>=0.19.0
sqlalchemy>=2.0.23

# Web UI and frontend
jinja2>=3.1.2
starlette>=0.32.0
python-multipart>=0.0.6

# Development and testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0
black>=23.11.0
flake8>=6.1.0
mypy>=1.7.1

# Utilities
click>=8.1.7
rich>=13.7.0
tabulate>=0.9.0
tqdm>=4.66.1

# JSON and data handling
orjson>=3.9.10
msgpack>=1.0.7

# Networking and protocols
dnspython>=2.4.2
certifi>=2023.11.17

# Optional: For advanced features
# Redis for caching
redis>=5.0.1

# For file watching and hot reload
watchdog>=3.0.0

# For advanced logging
colorlog>=6.8.0

# For metrics and monitoring
prometheus-client>=0.19.0

# For JWT tokens
pyjwt>=2.8.0

# For password hashing
argon2-cffi>=23.1.0

# For rate limiting
slowapi>=0.1.9

# For CORS
fastapi-cors>=0.0.6

# Platform-specific dependencies (Windows)
pywin32>=306; sys_platform == "win32"
wmi>=1.5.1; sys_platform == "win32"

# Platform-specific dependencies (Linux/Mac)
psutil>=5.9.6; sys_platform != "win32"
"""
    
    with open("requirements_enhanced.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Created enhanced requirements file: requirements_enhanced.txt")
    return True

def create_run_orchestrator_script():
    """Create a script to run the orchestrator."""
    script_content = '''#!/usr/bin/env python3
"""
Run the Universal MCP Orchestrator
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_web_server import main

if __name__ == "__main__":
    main()
'''
    
    # Create the script
    with open("run_orchestrator.py", "w") as f:
        f.write(script_content)
    
    # Make it executable on Unix systems
    if sys.platform != "win32":
        os.chmod("run_orchestrator.py", 0o755)
    
    print("✅ Created run_orchestrator.py script")
    return True

def create_orchestrator_config():
    """Create a sample configuration for the orchestrator."""
    config = {
        "description": "Universal MCP Orchestrator Configuration",
        "mcp_servers": {
            "web-search": {
                "name": "Web Search (No API Key)",
                "description": "Search the web using Google search with no API keys required",
                "repo": "https://github.com/pskill9/web-search",
                "auto_install": True
            },
            "filesystem": {
                "name": "File System Operations",
                "description": "Secure file operations with configurable access controls",
                "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "auto_install": False
            },
            "git": {
                "name": "Git Repository Operations",
                "description": "Tools to read, search, and manipulate Git repositories",
                "command": ["npx", "-y", "@modelcontextprotocol/server-git"],
                "auto_install": False
            }
        },
        "ui": {
            "title": "Universal MCP Orchestrator",
            "theme": "modern",
            "auto_select_defaults": True
        }
    }
    
    with open("orchestrator_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created orchestrator_config.json")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Universal MCP Orchestrator")
    print("=" * 50)
    
    # Check prerequisites
    if not check_node_npm():
        print("\n❌ Prerequisites not met. Please install Node.js and npm first.")
        sys.exit(1)
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Set up MCP servers
    setup_mcp_servers()
    
    # Create enhanced files
    create_enhanced_requirements()
    create_run_orchestrator_script()
    create_orchestrator_config()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Install enhanced requirements: pip install -r requirements_enhanced.txt")
    print("2. Configure your API keys in .env file")
    print("3. Run the orchestrator: python run_orchestrator.py")
    print("4. Open http://localhost:8080 in your browser")
    print("\n🌟 Features available:")
    print("   • Select AI models from OpenAI, Gemini, or Anthropic")
    print("   • Choose MCP servers (web search, filesystem, git)")
    print("   • Interactive chat with tool integration")
    print("   • Real-time web search without API keys")
    
    print("\n💡 Tip: Try asking about recent AI developments with web search enabled!")

if __name__ == "__main__":
    main()