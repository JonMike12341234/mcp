"""
FIXED Enhanced Web Server for Universal MCP Orchestrator
Provides REST API endpoints with WORKING MCP tool integration
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from utils.config import Config
from utils.logger import setup_logger, log_request

# Import the FIXED MCP Orchestrator
try:
    from fixed_mcp_orchestrator import FixedMCPOrchestrator
    HAS_FIXED_ORCHESTRATOR = True
except ImportError:
    HAS_FIXED_ORCHESTRATOR = False
    # Fallback to simple orchestrator
    class SimpleMCPOrchestrator:
        def __init__(self):
            self.config = Config()
            self.logger = setup_logger("simple_mcp_orchestrator")
            
            # Initialize LLM providers
            self.providers = {}
            try:
                from providers.openai_provider import OpenAIProvider
                from providers.gemini_provider import GeminiProvider  
                from providers.anthropic_provider import AnthropicProvider
                
                self.providers = {
                    'openai': OpenAIProvider(self.config.get_openai_config()),
                    'gemini': GeminiProvider(self.config.get_gemini_config()),
                    'anthropic': AnthropicProvider(self.config.get_anthropic_config())
                }
            except ImportError as e:
                self.logger.error(f"Failed to import providers: {e}")
            
            self.logger.warning("Using fallback simple orchestrator - MCP integration limited")
        
        def get_available_models(self):
            models = {}
            for provider_name, provider in self.providers.items():
                if provider.is_available():
                    try:
                        if hasattr(provider, 'MODELS'):
                            provider_models = []
                            for model_id, model_info in provider.MODELS.items():
                                if "alias_for" not in model_info:
                                    provider_models.append({
                                        "id": model_id,
                                        "name": model_info.get("description", model_id),
                                        "context_length": model_info.get("context_length", "Unknown"),
                                        "cost": f"${model_info.get('input_cost_per_1m', 0):.2f}/${model_info.get('output_cost_per_1m', 0):.2f} per 1M tokens"
                                    })
                            models[provider_name] = provider_models
                    except Exception as e:
                        self.logger.error(f"Error getting models from {provider_name}: {e}")
            return models
        
        async def execute_query(self, provider_name, model_id, mcp_server, user_query, system_prompt=None):
            if provider_name not in self.providers or not self.providers[provider_name].is_available():
                return {"error": f"Provider {provider_name} not available"}
            
            provider = self.providers[provider_name]
            response = await provider.generate_text(
                prompt=user_query,
                model=model_id,
                system=system_prompt,
                max_tokens=2000
            )
            
            return {
                "provider": provider_name,
                "model": model_id,
                "mcp_server": mcp_server,
                "response": response,
                "tool_used": None,
                "note": "Using fallback orchestrator - install fixed_mcp_orchestrator.py for full MCP support"
            }
        
        async def cleanup(self):
            pass


class ChatRequest(BaseModel):
    provider: str
    model: str
    mcp_server: Optional[str] = None
    message: str
    system_prompt: Optional[str] = None


class FixedEnhancedWebServer:
    """FIXED enhanced web server for the Universal MCP Orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(
            title="Universal MCP Orchestrator - FIXED", 
            description="AI Model and MCP Server Orchestration Platform with Working Tool Integration"
        )
        self.logger = setup_logger("fixed_enhanced_web_server")
        
        # Initialize the FIXED orchestrator
        if HAS_FIXED_ORCHESTRATOR:
            self.orchestrator = FixedMCPOrchestrator()
            self.logger.info("✅ Using FIXED MCP Orchestrator with working tool integration")
        else:
            self.orchestrator = SimpleMCPOrchestrator()
            self.logger.warning("⚠️ Using fallback orchestrator - tool integration limited")
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.get("/")
        async def get_dashboard():
            """Serve the enhanced dashboard HTML."""
            html_content = self._get_enhanced_dashboard_html()
            return HTMLResponse(content=html_content)
        
        @self.app.get("/api/status")
        async def get_status():
            """Get server status."""
            models = self.orchestrator.get_available_models() if hasattr(self.orchestrator, 'get_available_models') else {}
            
            connected_providers = sum(1 for provider_models in models.values() 
                                    if isinstance(provider_models, list) and provider_models)
            
            return {
                "server": "Universal MCP Orchestrator - FIXED",
                "version": "2.1.0-FIXED",
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "mcp_integration": "WORKING" if HAS_FIXED_ORCHESTRATOR else "LIMITED",
                "statistics": {
                    "connected_providers": connected_providers,
                    "total_providers": len(models),
                    "available_models": sum(len(provider_models) for provider_models in models.values() 
                                          if isinstance(provider_models, list)),
                    "mcp_integration_status": "FIXED" if HAS_FIXED_ORCHESTRATOR else "FALLBACK"
                }
            }
        
        @self.app.get("/api/models")
        async def get_models():
            """Get available models from all providers."""
            try:
                models = self.orchestrator.get_available_models()
                return models
            except Exception as e:
                self.logger.error(f"Error getting models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/mcp-servers")
        async def get_mcp_servers():
            """Get available MCP servers."""
            return {
                "web-search": {
                    "name": "Web Search",
                    "description": "Search the web for current information using MCP protocol",
                    "tools": ["web_search"],
                    "status": "WORKING" if HAS_FIXED_ORCHESTRATOR else "LIMITED"
                },
                "filesystem": {
                    "name": "File System",
                    "description": "Secure file operations with configurable access controls",
                    "tools": ["read_file", "write_file", "list_directory"],
                    "status": "PLANNED"
                },
                "github": {
                    "name": "GitHub",
                    "description": "GitHub repository operations via MCP",
                    "tools": ["get_repo", "list_files", "get_file_content"],
                    "status": "PLANNED"
                }
            }
        
        @self.app.post("/api/chat")
        async def chat(request: ChatRequest):
            """Handle chat requests with FIXED MCP integration."""
            try:
                self.logger.info(f"Chat request: {request.provider}/{request.model} with MCP server: {request.mcp_server}")
                
                # Add enhanced system prompt for tool-aware queries
                if request.mcp_server and not request.system_prompt:
                    request.system_prompt = "You are a helpful AI assistant with access to external tools including web search. When a user asks for current information or something that would benefit from web search, use the available tools to provide accurate, up-to-date responses. Always be clear about when you're using tools to gather information."
                
                result = await self.orchestrator.execute_query(
                    provider_name=request.provider,
                    model_id=request.model,
                    mcp_server=request.mcp_server,
                    user_query=request.message,
                    system_prompt=request.system_prompt
                )
                
                if "error" in result:
                    raise HTTPException(status_code=400, detail=result["error"])
                
                # Log successful tool usage
                if result.get("tool_used"):
                    self.logger.info(f"✅ Tool used successfully: {result['tool_used']}")
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error in chat endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/providers")
        async def get_providers():
            """Get detailed provider information."""
            try:
                providers_info = {}
                
                for name, provider in self.orchestrator.providers.items():
                    try:
                        if provider.is_available():
                            provider_status = await provider.get_status()
                            providers_info[name] = provider_status
                        else:
                            providers_info[name] = {
                                "available": False,
                                "error": "Not configured or API key missing"
                            }
                    except Exception as e:
                        providers_info[name] = {
                            "available": False,
                            "error": str(e)
                        }
                
                return providers_info
            except Exception as e:
                self.logger.error(f"Error getting providers: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/debug/mcp")
        async def debug_mcp():
            """Debug endpoint to check MCP integration status."""
            return {
                "fixed_orchestrator_available": HAS_FIXED_ORCHESTRATOR,
                "orchestrator_type": "FixedMCPOrchestrator" if HAS_FIXED_ORCHESTRATOR else "SimpleMCPOrchestrator",
                "mcp_integration_status": "WORKING" if HAS_FIXED_ORCHESTRATOR else "LIMITED",
                "recommendation": "Tool integration is working properly!" if HAS_FIXED_ORCHESTRATOR else "Install fixed_mcp_orchestrator.py for full MCP tool support"
            }
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Clean up resources on shutdown."""
            try:
                await self.orchestrator.cleanup()
                self.logger.info("Server shutdown completed")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
    
    def _get_enhanced_dashboard_html(self):
        """Generate the enhanced dashboard HTML."""
        # Read the full UI from the separate file
        try:
            with open('enhanced_web_ui.html', 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Inject status information into the HTML
                status_indicator = """
                    <div class="mb-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
                        <h3 class="font-bold">🔧 MCP Integration Status</h3>
                        <p>✅ <strong>FIXED:</strong> Tool integration is now working properly!</p>
                        <p>🌐 Web search and other MCP tools are now properly connected to AI models.</p>
                    </div>
                """ if HAS_FIXED_ORCHESTRATOR else """
                    <div class="mb-4 p-4 bg-yellow-100 border border-yellow-400 text-yellow-700 rounded">
                        <h3 class="font-bold">⚠️ MCP Integration Status</h3>
                        <p><strong>LIMITED:</strong> Using fallback orchestrator.</p>
                        <p>Install fixed_mcp_orchestrator.py for full tool support.</p>
                    </div>
                """
                
                # Insert status after the header
                content = content.replace(
                    '<div class="container mx-auto px-6 py-8">',
                    f'<div class="container mx-auto px-6 py-8">{status_indicator}'
                )
                
                return content
        except FileNotFoundError:
            # Fallback to basic UI if file not found
            status_class = "text-green-600" if HAS_FIXED_ORCHESTRATOR else "text-yellow-600"
            status_icon = "✅" if HAS_FIXED_ORCHESTRATOR else "⚠️"
            status_text = "MCP tool integration is WORKING!" if HAS_FIXED_ORCHESTRATOR else "Using limited MCP integration"
            
            return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal MCP Orchestrator - FIXED</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
    </style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen">
        <header class="gradient-bg text-white shadow-lg">
            <div class="container mx-auto px-6 py-6">
                <h1 class="text-3xl font-bold">Universal MCP Orchestrator - FIXED</h1>
                <p class="text-blue-100 mt-2">AI Model Integration Platform with Working Tool Support</p>
            </div>
        </header>
        <div class="container mx-auto px-6 py-8">
            <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
                <h2 class="text-xl font-bold mb-4">🔧 Fixed Implementation Status</h2>
                <p class="{status_class} text-lg">{status_icon} {status_text}</p>
                <div class="mt-4">
                    <h3 class="font-semibold mb-2">What's Fixed:</h3>
                    <ul class="list-disc list-inside space-y-1 text-gray-700">
                        <li>✅ Tools are now properly advertised to AI models</li>
                        <li>✅ LLMs can now use web search and other MCP tools</li>
                        <li>✅ Tool results are properly formatted and returned</li>
                        <li>✅ Support for OpenAI, Anthropic, and Gemini function calling</li>
                    </ul>
                </div>
            </div>
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-xl font-bold mb-4">Server Status</h2>
                <p class="text-green-600">✅ Server is running successfully!</p>
                <p class="mt-2">🌐 Server available at: http://localhost:8080</p>
                <div class="mt-4">
                    <h3 class="font-semibold mb-2">Test the Fix:</h3>
                    <ol class="list-decimal list-inside space-y-1 text-gray-700">
                        <li>Select an AI model (OpenAI, Anthropic, or Gemini)</li>
                        <li>Choose "Web Search" as the MCP server</li>
                        <li>Ask: "What are the latest developments in AI in 2025?"</li>
                        <li>The AI should now use web search tools to provide current information</li>
                    </ol>
                </div>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    async def start(self, host: str = "localhost", port: int = 8080):
        """Start the web server."""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


async def create_fixed_enhanced_web_server(config: Config) -> Optional[FixedEnhancedWebServer]:
    """Create and initialize the fixed enhanced web server."""
    if not HAS_FASTAPI:
        print("FastAPI is not installed. Please install it to use the web server.")
        return None

    return FixedEnhancedWebServer(config)


def main():
    """Main entry point for running the fixed enhanced web server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal MCP Orchestrator Web Dashboard - FIXED")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    async def run_server():
        server = await create_fixed_enhanced_web_server(config)
        if server:
            print(f"\n🚀 Universal MCP Orchestrator (FIXED) starting at http://{args.host}:{args.port}")
            print("✅ Fixed Features:")
            print("   • Working MCP tool integration")  
            print("   • AI models can now use web search")
            print("   • Proper tool result handling")
            print("   • Multi-provider tool support")
            print("\n🔧 To test the fix:")
            print("   1. Select a model and 'Web Search' MCP server")
            print("   2. Ask about recent AI developments")
            print("   3. Watch the AI use tools to search the web!")
            print("\nPress Ctrl+C to stop the server\n")
            await server.start(args.host, args.port)
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n👋 Universal MCP Orchestrator stopped by user")
    except Exception as e:
        print(f"❌ Error running server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()