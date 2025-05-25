#!/usr/bin/env python3
"""
Universal MCP Server
A comprehensive Model Context Protocol server supporting OpenAI, Gemini, and Anthropic Claude models.
"""

import asyncio
import os
import sys
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import argparse
import logging
from datetime import datetime

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource,
    CallToolRequest,
    ReadResourceRequest,
    ListResourcesRequest,
    ListToolsRequest,
)

# Provider-specific imports
from providers.openai_provider import OpenAIProvider
from providers.gemini_provider import GeminiProvider  
from providers.anthropic_provider import AnthropicProvider
from utils.config import Config
from utils.logger import setup_logger
from utils.security import SecurityManager

class UniversalMCPServer:
    """Universal MCP Server supporting multiple LLM providers."""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger(__name__)
        self.security = SecurityManager()
        
        # Initialize providers
        self.providers = {
            'openai': OpenAIProvider(self.config.get_openai_config()),
            'gemini': GeminiProvider(self.config.get_gemini_config()),
            'anthropic': AnthropicProvider(self.config.get_anthropic_config())
        }
        
        # Initialize MCP server
        self.server = Server("universal-mcp-server")
        self._setup_handlers()
        
        self.logger.info("Universal MCP Server initialized successfully")
    
    def _setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools from all providers."""
            tools = []
            
            # Core tools available for all providers
            core_tools = [
                Tool(
                    name="generate_text",
                    description="Generate text using specified LLM provider",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string", 
                                "enum": ["openai", "gemini", "anthropic"],
                                "description": "LLM provider to use"
                            },
                            "model": {
                                "type": "string",
                                "description": "Specific model to use (optional, uses default if not specified)"
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Text prompt to generate from"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "default": 1000,
                                "description": "Maximum tokens to generate"
                            },
                            "temperature": {
                                "type": "number",
                                "default": 0.7,
                                "description": "Temperature for generation (0.0-2.0)"
                            }
                        },
                        "required": ["provider", "prompt"]
                    }
                ),
                Tool(
                    name="list_models",
                    description="List available models for each provider",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "enum": ["openai", "gemini", "anthropic", "all"],
                                "default": "all",
                                "description": "Provider to list models for"
                            }
                        }
                    }
                ),
                Tool(
                    name="compare_responses",
                    description="Compare responses from multiple providers for the same prompt",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Prompt to compare across providers"
                            },
                            "providers": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["openai", "gemini", "anthropic"]},
                                "default": ["openai", "gemini", "anthropic"],
                                "description": "Providers to compare"
                            },
                            "models": {
                                "type": "object",
                                "description": "Specific models for each provider (optional)"
                            }
                        },
                        "required": ["prompt"]
                    }
                )
            ]
            
            tools.extend(core_tools)
            
            # Add provider-specific tools
            for provider_name, provider in self.providers.items():
                if provider.is_available():
                    try:
                        provider_tools = await provider.get_available_tools()
                        for tool in provider_tools:
                            tool.name = f"{provider_name}_{tool.name}"
                            tools.append(tool)
                    except Exception as e:
                        self.logger.error(f"Error getting tools from {provider_name}: {e}")
            
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                self.logger.info(f"Calling tool: {name} with arguments: {arguments}")
                
                if name == "generate_text":
                    return await self._handle_generate_text(arguments)
                elif name == "list_models":
                    return await self._handle_list_models(arguments)
                elif name == "compare_responses":
                    return await self._handle_compare_responses(arguments)
                else:
                    # Handle provider-specific tools
                    for provider_name, provider in self.providers.items():
                        if name.startswith(f"{provider_name}_"):
                            tool_name = name[len(f"{provider_name}_"):]
                            return await provider.call_tool(tool_name, arguments)
                    
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                self.logger.error(f"Error calling tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources from all providers."""
            resources = []
            
            # Core resources
            core_resources = [
                Resource(
                    uri="server_config",
                    description="Configuration of the Universal MCP Server",
                    mimeType="application/json"
                ),
                Resource(
                    uri="provider_status",
                    description="Status of the configured LLM providers",
                    mimeType="application/json"
                ),
                Resource(
                    uri="recent_logs",
                    description="Recent logs from the server",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="usage_metrics",
                    description="Usage metrics for the server",
                    mimeType="application/json"
                )
            ]
            
            resources.extend(core_resources)
            
            # Add provider-specific resources
            for provider_name, provider in self.providers.items():
                if provider.is_available():
                    try:
                        provider_resources = await provider.get_available_resources()
                        for resource in provider_resources:
                            resource.uri = f"{provider_name}_{resource.uri}"
                            resources.append(resource)
                    except Exception as e:
                        self.logger.error(f"Error getting resources from {provider_name}: {e}")
            
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Get a specific resource by URI."""
            try:
                self.logger.info(f"Fetching resource: {uri}")
                
                # Handle core resources
                if uri == "server_config":
                    config = await self._get_server_config()
                    return json.dumps(config, indent=2)
                elif uri == "provider_status":
                    status = await self._get_provider_status()
                    return json.dumps(status, indent=2)
                elif uri == "recent_logs":
                    logs = await self._get_recent_logs()
                    return logs
                elif uri == "usage_metrics":
                    metrics = await self._get_usage_metrics()
                    return json.dumps(metrics, indent=2)
                
                # Handle provider-specific resources
                if "_" in uri:
                    provider_name, resource_uri = uri.split("_", 1)
                    if provider_name in self.providers:
                        provider = self.providers[provider_name]
                        if provider.is_available():
                            return await provider.get_resource(resource_uri)
                
                return f"Resource not found: {uri}"
            
            except Exception as e:
                self.logger.error(f"Error fetching resource {uri}: {str(e)}")
                return f"Error: {str(e)}"
    
    async def _handle_generate_text(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle text generation requests."""
        provider_name = arguments.get("provider")
        prompt = arguments.get("prompt")
        model = arguments.get("model")
        max_tokens = arguments.get("max_tokens", 1000)
        temperature = arguments.get("temperature", 0.7)
        
        if provider_name not in self.providers:
            return [TextContent(type="text", text=f"Unknown provider: {provider_name}")]
        
        provider = self.providers[provider_name]
        if not provider.is_available():
            return [TextContent(type="text", text=f"Provider {provider_name} is not available")]
        
        try:
            response = await provider.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return [TextContent(type="text", text=response)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating text: {str(e)}")]
    
    async def _handle_list_models(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle model listing requests."""
        provider_filter = arguments.get("provider", "all")
        
        models_info = {}
        
        for provider_name, provider in self.providers.items():
            if provider_filter == "all" or provider_filter == provider_name:
                if provider.is_available():
                    try:
                        models_info[provider_name] = await provider.list_models()
                    except Exception as e:
                        models_info[provider_name] = f"Error: {str(e)}"
                else:
                    models_info[provider_name] = "Provider not available"
        
        return [TextContent(type="text", text=json.dumps(models_info, indent=2))]
    
    async def _handle_compare_responses(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle response comparison requests."""
        prompt = arguments.get("prompt")
        providers_to_compare = arguments.get("providers", ["openai", "gemini", "anthropic"])
        models = arguments.get("models", {})
        
        results = {}
        
        for provider_name in providers_to_compare:
            if provider_name in self.providers and self.providers[provider_name].is_available():
                try:
                    model = models.get(provider_name)
                    response = await self.providers[provider_name].generate_text(
                        prompt=prompt,
                        model=model
                    )
                    results[provider_name] = {
                        "response": response,
                        "model_used": model or "default",
                        "status": "success"
                    }
                except Exception as e:
                    results[provider_name] = {
                        "error": str(e),
                        "status": "error"
                    }
            else:
                results[provider_name] = {
                    "error": "Provider not available",
                    "status": "unavailable"
                }
        
        comparison_text = f"Prompt: {prompt}\n\n"
        for provider, result in results.items():
            comparison_text += f"--- {provider.upper()} ---\n"
            if result["status"] == "success":
                comparison_text += f"Model: {result['model_used']}\n"
                comparison_text += f"Response: {result['response']}\n\n"
            else:
                comparison_text += f"Error: {result.get('error', 'Unknown error')}\n\n"
        
        return [TextContent(type="text", text=comparison_text)]
    
    async def _get_provider_status(self) -> Dict[str, Any]:
        """Get provider status information."""
        status_info = {}
        
        for provider_name, provider in self.providers.items():
            try:
                if provider.is_available():
                    status = await provider.get_status()
                    status_info[provider_name] = status
                else:
                    status_info[provider_name] = {"available": False, "error": "Not configured"}
            except Exception as e:
                status_info[provider_name] = {"available": False, "error": str(e)}
        
        return status_info
    
    async def _get_server_config(self) -> Dict[str, Any]:
        """Get the server configuration."""
        return {
            "server_version": "1.0.0",
            "supported_providers": list(self.providers.keys()),
            "available_providers": [name for name, provider in self.providers.items() if provider.is_available()],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_recent_logs(self) -> str:
        """Get the recent logs from the server."""
        log_file = Path("logs/server.log")
        if not log_file.exists():
            return "No logs found"
        
        try:
            with open(log_file, "r", encoding='utf-8') as f:
                lines = f.readlines()
                # Return last 50 lines
                return ''.join(lines[-50:])
        except Exception as e:
            return f"Error reading logs: {str(e)}"
    
    async def _get_usage_metrics(self) -> Dict[str, Any]:
        """Get the usage metrics for the server."""
        # Placeholder for actual metrics collection logic
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "uptime": "0 minutes",
            "last_updated": datetime.now().isoformat()
        }
    
    async def run(self):
        """Run the MCP server."""
        self.logger.info("Starting Universal MCP Server...")
        
        # Check provider availability
        available_providers = []
        for name, provider in self.providers.items():
            if provider.is_available():
                available_providers.append(name)
                self.logger.info(f"Provider {name} is available")
            else:
                self.logger.warning(f"Provider {name} is not available")
        
        if not available_providers:
            self.logger.error("No providers are available. Please check your configuration.")
            sys.exit(1)
        
        self.logger.info(f"Server running with providers: {', '.join(available_providers)}")
        
        # Run the MCP server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Universal MCP Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--port", type=int, help="Port for HTTP transport (optional)")
    
    args = parser.parse_args()
    
    # Set up logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        server = UniversalMCPServer()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()