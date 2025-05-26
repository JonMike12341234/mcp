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
        print("🚀 Starting Universal MCP Orchestrator (FIXED)...")
        
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("⚠️  Warning: .env file not found!")
            print("Please create .env file with your API keys:")
            print("OPENAI_API_KEY=your_key_here")
            print("GEMINI_API_KEY=your_key_here") 
            print("ANTHROPIC_API_KEY=your_key_here")
            print()
        
        # Import and run the FIXED server
        try:
            print("✅ Loading FIXED enhanced web server...")
            
            from fixed_enhanced_web_server import main as server_main
            
            print("✅ Starting FIXED server with working tool integration...")
            print("🌐 Server will be available at: http://localhost:8080")
            print("🔧 MCP tool integration is now WORKING!")
            print("Press Ctrl+C to stop the server")
            print()
            
            server_main()
            
        except ImportError as e:
            print(f"❌ Import Error: {e}")
            print("\n💡 Make sure you have saved both fixed files:")
            print("  • fixed_mcp_orchestrator.py")
            print("  • fixed_enhanced_web_server.py")
            input("Press Enter to exit...")
            sys.exit(1)
        
        except Exception as e:
            print(f"❌ Server Error: {e}")
            traceback.print_exc()
            input("Press Enter to exit...")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Universal MCP Orchestrator stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()