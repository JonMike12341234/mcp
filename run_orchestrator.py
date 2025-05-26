#!/usr/bin/env python3
"""
Run the Universal MCP Orchestrator
"""

import sys
import traceback

if __name__ == "__main__":
    try:
        print("🚀 Starting Universal MCP Orchestrator...")
        from enhanced_web_server import main
        print("✅ Imports successful, starting server...")
        main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 This usually means missing dependencies.")
        print("Run: pip install fastapi uvicorn mcp")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"📍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)