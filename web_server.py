"""
Web Server for Universal MCP Server Dashboard
Provides REST API endpoints and serves the web UI
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
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from utils.config import Config
from utils.logger import setup_logger, log_request
from utils.security import SecurityManager
from providers.openai_provider import OpenAIProvider
from providers.gemini_provider import GeminiProvider
from providers.anthropic_provider import AnthropicProvider

class WebServer:
    """Web server for the Universal MCP Server dashboard and API."""
    
    def __init__(self, config: Config, providers: Dict[str, Any]):
        self.config = config
        self.providers = providers
        self.app = FastAPI(title="Universal MCP Server", description="Multi-Provider LLM Integration Dashboard")
        self.logger = logging.getLogger(__name__)
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
            """Serve the main dashboard HTML."""
            html_content = self._get_dashboard_html()
            return HTMLResponse(content=html_content)
        
        @self.app.get("/api/status")
        async def get_status():
            """Get server status."""
            status = {
                "server": "Universal MCP Server",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "uptime": "Unknown",
                "providers": {}
            }
            
            # Get provider status
            for name, provider in self.providers.items():
                try:
                    if provider.is_available():
                        status["providers"][name] = {
                            "available": True,
                            "status": "connected"
                        }
                    else:
                        status["providers"][name] = {
                            "available": False,
                            "status": "not configured"
                        }
                except Exception as e:
                    status["providers"][name] = {
                        "available": False,
                        "status": f"error: {str(e)}"
                    }
            
            return status
        
        @self.app.get("/api/providers")
        async def get_providers():
            """Get detailed provider information."""
            providers_info = {}
            
            for name, provider in self.providers.items():
                try:
                    if provider.is_available():
                        # Try to get detailed status
                        try:
                            provider_status = await provider.get_status()
                            providers_info[name] = provider_status
                        except Exception as e:
                            providers_info[name] = {
                                "available": True,
                                "error": f"Status check failed: {str(e)}"
                            }
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
        
        @self.app.get("/api/models")
        async def get_models():
            """Get available models from all providers."""
            models_info = {}
            
            for name, provider in self.providers.items():
                try:
                    if provider.is_available():
                        try:
                            models = await provider.list_models()
                            models_info[name] = models
                        except Exception as e:
                            models_info[name] = {"error": str(e)}
                    else:
                        models_info[name] = {"error": "Provider not available"}
                except Exception as e:
                    models_info[name] = {"error": str(e)}
            
            return models_info
        
        @self.app.get("/resource/{uri}")
        async def get_resource(uri: str):
            """Get a resource by URI."""
            for provider in self.providers.values():
                if provider.is_available():
                    try:
                        resource = await provider.get_resource(uri)
                        if resource:
                            return {"content": resource}
                    except Exception as e:
                        self.logger.error(f"Error getting resource {uri} from provider: {e}")
            
            raise HTTPException(status_code=404, detail=f"Resource {uri} not found")
    
    def _get_dashboard_html(self):
        """Generate dashboard HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal MCP Server - Dashboard</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5; color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px;
            text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; }
        .header p { margin: 10px 0 0; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: white; padding: 25px; border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #667eea;
        }
        .card h3 { margin: 0 0 15px; color: #333; font-size: 1.3em; }
        .status { padding: 8px 16px; border-radius: 20px; font-size: 0.9em; font-weight: 500; }
        .status.available { background: #d4edda; color: #155724; }
        .status.unavailable { background: #f8d7da; color: #721c24; }
        .status.loading { background: #fff3cd; color: #856404; }
        .refresh-btn { 
            background: #667eea; color: white; border: none; padding: 10px 20px;
            border-radius: 5px; cursor: pointer; margin: 10px 0;
        }
        .refresh-btn:hover { background: #5a6fd8; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }
        .info-item { padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .info-label { font-weight: 600; color: #666; font-size: 0.9em; }
        .info-value { color: #333; margin-top: 5px; }
        .loading { text-align: center; padding: 20px; color: #666; }
        .error { color: #721c24; background: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .footer { text-align: center; margin-top: 40px; color: #666; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Universal MCP Server</h1>
            <p>Multi-Provider LLM Integration Dashboard</p>
            <p id="current-time"></p>
        </div>
        
        <div id="server-status" class="card">
            <h3>Server Status</h3>
            <div class="loading">Loading server status...</div>
        </div>
        
        <div class="grid" id="providers-grid">
            <div class="loading">Loading provider information...</div>
        </div>
        
        <div class="footer">
            <p>Universal MCP Server Dashboard • <button class="refresh-btn" onclick="refreshAll()">Refresh All</button></p>
            <p>MCP Server running on stdio transport • Web dashboard for monitoring only</p>
        </div>
    </div>

    <script>
        // Update current time
        function updateTime() {
            document.getElementById('current-time').textContent = new Date().toLocaleString();
        }
        updateTime();
        setInterval(updateTime, 1000);

        // Load server status
        async function loadServerStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                const statusHtml = `
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">Server Version</div>
                            <div class="info-value">${data.version}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Status</div>
                            <div class="info-value">
                                <span class="status available">${data.status}</span>
                            </div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Last Updated</div>
                            <div class="info-value">${new Date(data.timestamp).toLocaleString()}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Available Providers</div>
                            <div class="info-value">${Object.values(data.providers).filter(p => p.available).length} / ${Object.keys(data.providers).length}</div>
                        </div>
                    </div>
                `;
                
                document.getElementById('server-status').innerHTML = '<h3>Server Status</h3>' + statusHtml;
            } catch (error) {
                document.getElementById('server-status').innerHTML = 
                    '<h3>Server Status</h3><div class="error">Failed to load server status: ' + error.message + '</div>';
            }
        }

        // Load providers information
        async function loadProviders() {
            try {
                const response = await fetch('/api/providers');
                const data = await response.json();
                
                let html = '';
                for (const [name, info] of Object.entries(data)) {
                    const statusClass = info.available ? 'available' : 'unavailable';
                    const statusText = info.available ? 'Available' : 'Not Available';
                    
                    // Handle specific error types
                    let errorDisplay = '';
                    if (info.error) {
                        if (info.error_type === 'billing') {
                            errorDisplay = `<div class="error"><strong>💳 Billing Issue:</strong><br>${info.error}</div>`;
                        } else if (info.error_type === 'authentication') {
                            errorDisplay = `<div class="error"><strong>🔑 Authentication Issue:</strong><br>${info.error}</div>`;
                        } else {
                            errorDisplay = `<div class="error"><strong>❌ Error:</strong><br>${info.error}</div>`;
                        }
                    }
                    
                    html += `
                        <div class="card">
                            <h3>${name.charAt(0).toUpperCase() + name.slice(1)} Provider</h3>
                            <div class="status ${statusClass}">${statusText}</div>
                            <div class="info-grid">
                                ${info.default_model ? `
                                    <div class="info-item">
                                        <div class="info-label">Default Model</div>
                                        <div class="info-value">${info.default_model}</div>
                                    </div>
                                ` : ''}
                                ${info.request_count !== undefined ? `
                                    <div class="info-item">
                                        <div class="info-label">Requests</div>
                                        <div class="info-value">${info.request_count}</div>
                                    </div>
                                ` : ''}
                                ${info.total_cost_usd !== undefined ? `
                                    <div class="info-item">
                                        <div class="info-label">Total Cost</div>
                                        <div class="info-value">$${info.total_cost_usd.toFixed(4)}</div>
                                    </div>
                                ` : ''}
                                ${errorDisplay ? `
                                    <div class="info-item" style="grid-column: 1 / -1;">
                                        ${errorDisplay}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }
                
                document.getElementById('providers-grid').innerHTML = html;
            } catch (error) {
                document.getElementById('providers-grid').innerHTML = 
                    '<div class="error">Failed to load providers: ' + error.message + '</div>';
            }
        }

        // Refresh all data
        function refreshAll() {
            loadServerStatus();
            loadProviders();
        }

        // Initial load
        refreshAll();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshAll, 30000);
    </script>
</body>
</html>'''
    
    def is_available(self) -> bool:
        """Check if the web server is available."""
        return True
    
    async def start(self, host: str = "localhost", port: int = 8080):
        """Start the web server."""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    
    def run_sync(self, host: str = "localhost", port: int = 8080):
        """Run the web server synchronously (blocking)."""
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(self.start(host, port))

async def create_web_server(config: Config) -> Optional[WebServer]:
    """Create and initialize the web server with providers."""
    if not HAS_FASTAPI:
        print("FastAPI is not installed. Please install it to use the web server.")
        return None
    
    # Initialize providers
    providers = {}
    
    # OpenAI provider
    openai_config = config.get_openai_config()
    if openai_config.get("api_key"):
        providers["openai"] = OpenAIProvider(openai_config)
    
    # Gemini provider
    gemini_config = config.get_gemini_config()
    if gemini_config.get("api_key"):
        providers["gemini"] = GeminiProvider(gemini_config)
    
    # Anthropic provider
    anthropic_config = config.get_anthropic_config()
    if anthropic_config.get("api_key"):
        providers["anthropic"] = AnthropicProvider(anthropic_config)
    
    if not providers:
        print("No valid providers found in the configuration.")
        return None

    return WebServer(config, providers)

def main():
    """Main entry point for running the web server standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal MCP Server Web Dashboard")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    async def run_server():
        server = await create_web_server(config)
        if server:
            await server.start(args.host, args.port)
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nWeb server stopped by user")
    except Exception as e:
        print(f"Error running web server: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()