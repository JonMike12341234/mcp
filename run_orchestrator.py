#!/usr/bin/env python3
"""
Run the Universal MCP Orchestrator - FIXED VERSION
"""

import sys
import traceback
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for running the orchestrator."""
    try:
        print("🚀 Starting Universal MCP Orchestrator...")
        
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("⚠️  Warning: .env file not found!")
            print("Please create .env file with your API keys:")
            print("OPENAI_API_KEY=your_key_here")
            print("GEMINI_API_KEY=your_key_here") 
            print("ANTHROPIC_API_KEY=your_key_here")
            print()
            print("You need at least ONE API key to use the orchestrator.")
            print("Continuing anyway...")
            print()
        
        # Try to import and run the server
        try:
            print("✅ Loading enhanced web server...")
            
            # Import the fixed server
            from enhanced_web_server import main as server_main
            
            print("✅ Starting server...")
            server_main()
            
        except ImportError as e:
            print(f"❌ Import Error: {e}")
            print()
            print("💡 This usually means missing dependencies.")
            print("Please run:")
            print("  pip install fastapi uvicorn pydantic python-dotenv")
            print("  pip install openai google-generativeai anthropic")  
            print()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Universal MCP Orchestrator stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("📍 Full traceback:")
        traceback.print_exc()
        print()
        print("💡 Common issues:")
        print("  • Missing API keys in .env file")
        print("  • Missing Python dependencies")
        print("  • Port 8080 already in use")
        print("  • Permission issues")
        sys.exit(1)

if __name__ == "__main__":
    main()