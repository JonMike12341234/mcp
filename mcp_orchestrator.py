#!/usr/bin/env python3
"""
Enhanced Universal MCP Server - MCP Orchestrator
Allows users to select LLM models and MCP servers for orchestrated AI interactions
"""

import asyncio
import json
import subprocess
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import signal

from mcp.client import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest, ListToolsRequest

from providers.openai_provider import OpenAIProvider
from providers.gemini_provider import GeminiProvider  
from providers.anthropic_provider import AnthropicProvider
from utils.config import Config
from utils.logger import setup_logger

class MCPServerManager:
    """Manages external MCP servers that can be connected to LLM providers."""
    
    def __init__(self):
        self.logger = setup_logger("mcp_server_manager")
        self.available_servers = {
            "web-search": {
                "name": "Web Search (No API Key)",
                "description": "Search the web using Google search with no API keys required",
                "repo": "https://github.com/pskill9/web-search",
                "command": ["node"],
                "args": [],  # Will be set after installation
                "tools": ["search"],
                "installed": False,
                "process": None,
                "session": None
            },
            "filesystem": {
                "name": "File System Operations",
                "description": "Secure file operations with configurable access controls",
                "command": ["npx"],
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "tools": ["read_file", "write_file", "create_directory", "list_directory"],
                "installed": True,
                "process": None,
                "session": None
            },
            "git": {
                "name": "Git Repository Operations", 
                "description": "Tools to read, search, and manipulate Git repositories",
                "command": ["npx"],
                "args": ["-y", "@modelcontextprotocol/server-git"],
                "tools": ["git_log", "git_diff", "git_show", "search_files"],
                "installed": True,
                "process": None,
                "session": None
            }
        }
        self.active_connections = {}
    
    async def install_web_search_server(self) -> bool:
        """Install the web search MCP server."""
        try:
            self.logger.info("Installing web search MCP server...")
            
            # Create a temporary directory for the server
            temp_dir = Path(tempfile.mkdtemp(prefix="mcp_websearch_"))
            
            # Clone the repository
            clone_result = subprocess.run([
                "git", "clone", "https://github.com/pskill9/web-search.git", str(temp_dir)
            ], capture_output=True, text=True)
            
            if clone_result.returncode != 0:
                self.logger.error(f"Failed to clone web search repo: {clone_result.stderr}")
                return False
            
            # Install dependencies
            install_result = subprocess.run([
                "npm", "install"
            ], cwd=temp_dir, capture_output=True, text=True)
            
            if install_result.returncode != 0:
                self.logger.error(f"Failed to install dependencies: {install_result.stderr}")
                return False
            
            # Build the project
            build_result = subprocess.run([
                "npm", "run", "build"
            ], cwd=temp_dir, capture_output=True, text=True)
            
            if build_result.returncode != 0:
                self.logger.error(f"Failed to build project: {build_result.stderr}")
                return False
            
            # Update the server configuration
            build_path = temp_dir / "build" / "index.js"
            if build_path.exists():
                self.available_servers["web-search"]["args"] = [str(build_path)]
                self.available_servers["web-search"]["installed"] = True
                self.logger.info("Web search MCP server installed successfully")
                return True
            else:
                self.logger.error("Build file not found after compilation")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing web search server: {e}")
            return False
    
    async def start_mcp_server(self, server_id: str) -> Optional[ClientSession]:
        """Start an MCP server and return a client session."""
        if server_id not in self.available_servers:
            self.logger.error(f"Unknown server: {server_id}")
            return None
        
        server_config = self.available_servers[server_id]
        
        # Install web search server if needed
        if server_id == "web-search" and not server_config["installed"]:
            if not await self.install_web_search_server():
                return None
        
        try:
            self.logger.info(f"Starting MCP server: {server_config['name']}")
            
            # Start the server process
            command = server_config["command"] + server_config["args"]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False  # Use binary mode for MCP
            )
            
            # Create stdio client session
            session = await stdio_client(process.stdin, process.stdout)
            await session.initialize()
            
            # Store the process and session
            server_config["process"] = process
            server_config["session"] = session
            self.active_connections[server_id] = session
            
            self.logger.info(f"Successfully connected to {server_config['name']}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server {server_id}: {e}")
            return None
    
    async def stop_mcp_server(self, server_id: str):
        """Stop an MCP server."""
        if server_id not in self.available_servers:
            return
        
        server_config = self.available_servers[server_id]
        
        # Close session
        if server_config["session"]:
            try:
                await server_config["session"].close()
            except:
                pass
            server_config["session"] = None
        
        # Terminate process
        if server_config["process"]:
            try:
                server_config["process"].terminate()
                try:
                    server_config["process"].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_config["process"].kill()
            except:
                pass
            server_config["process"] = None
        
        # Remove from active connections
        self.active_connections.pop(server_id, None)
        
        self.logger.info(f"Stopped MCP server: {server_config['name']}")
    
    async def get_available_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """Get available tools from an MCP server."""
        if server_id not in self.active_connections:
            return []
        
        try:
            session = self.active_connections[server_id]
            result = await session.list_tools()
            return [tool.model_dump() for tool in result.tools]
        except Exception as e:
            self.logger.error(f"Error getting tools from {server_id}: {e}")
            return []
    
    async def call_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on an MCP server."""
        if server_id not in self.active_connections:
            return {"error": "Server not connected"}
        
        try:
            session = self.active_connections[server_id]
            result = await session.call_tool(tool_name, arguments)
            return result.content
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name} on {server_id}: {e}")
            return {"error": str(e)}
    
    def get_available_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available MCP servers."""
        return {
            server_id: {
                "name": config["name"],
                "description": config["description"],
                "tools": config["tools"],
                "installed": config["installed"],
                "connected": server_id in self.active_connections
            }
            for server_id, config in self.available_servers.items()
        }
    
    async def cleanup(self):
        """Clean up all MCP server connections."""
        for server_id in list(self.active_connections.keys()):
            await self.stop_mcp_server(server_id)


class MCPOrchestrator:
    """Main orchestrator that connects LLM providers with MCP servers."""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger("mcp_orchestrator")
        
        # Initialize LLM providers
        self.providers = {
            'openai': OpenAIProvider(self.config.get_openai_config()),
            'gemini': GeminiProvider(self.config.get_gemini_config()),
            'anthropic': AnthropicProvider(self.config.get_anthropic_config())
        }
        
        # Initialize MCP server manager
        self.mcp_manager = MCPServerManager()
        
        self.logger.info("MCP Orchestrator initialized")
    
    def get_available_models(self) -> Dict[str, List[Dict[str, str]]]:
        """Get available models from all providers."""
        models = {}
        
        for provider_name, provider in self.providers.items():
            if provider.is_available():
                try:
                    # Get models from provider
                    if hasattr(provider, 'MODELS'):
                        provider_models = []
                        for model_id, model_info in provider.MODELS.items():
                            if "alias_for" not in model_info:  # Skip aliases
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
    
    async def execute_query(
        self, 
        provider_name: str, 
        model_id: str, 
        mcp_server: Optional[str],
        user_query: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a query using the selected provider and MCP server."""
        try:
            # Validate provider
            if provider_name not in self.providers or not self.providers[provider_name].is_available():
                return {"error": f"Provider {provider_name} not available"}
            
            provider = self.providers[provider_name]
            
            # If MCP server is specified, connect and get tools
            mcp_tools = []
            mcp_session = None
            
            if mcp_server:
                mcp_session = await self.mcp_manager.start_mcp_server(mcp_server)
                if mcp_session:
                    mcp_tools = await self.mcp_manager.get_available_tools(mcp_server)
                    self.logger.info(f"Connected to MCP server {mcp_server} with {len(mcp_tools)} tools")
                else:
                    return {"error": f"Failed to connect to MCP server {mcp_server}"}
            
            # Prepare the enhanced prompt
            enhanced_prompt = user_query
            
            if mcp_tools:
                # Add tool descriptions to the prompt
                tool_descriptions = []
                for tool in mcp_tools:
                    tool_desc = f"- {tool['name']}: {tool.get('description', 'No description')}"
                    tool_descriptions.append(tool_desc)
                
                enhanced_prompt = f"""You have access to the following tools:
{chr(10).join(tool_descriptions)}

If the user's query would benefit from using any of these tools, please indicate which tool you would like to use and with what parameters.

User query: {user_query}

If you need to use a tool, respond with: USE_TOOL: tool_name with parameters: {{parameters}}
Otherwise, respond normally to the user's query."""
            
            # Generate initial response
            response = await provider.generate_text(
                prompt=enhanced_prompt,
                model=model_id,
                system=system_prompt,
                max_tokens=2000
            )
            
            # Check if the model wants to use a tool
            if mcp_session and "USE_TOOL:" in response:
                try:
                    # Parse tool usage (basic parsing - in production you'd want more robust parsing)
                    parts = response.split("USE_TOOL:")
                    tool_part = parts[1].strip()
                    
                    if "with parameters:" in tool_part:
                        tool_name = tool_part.split("with parameters:")[0].strip()
                        params_str = tool_part.split("with parameters:")[1].strip()
                        
                        # For web search, extract query
                        if tool_name == "search" and mcp_server == "web-search":
                            # Extract search query from user's original message
                            search_query = user_query  # Simplified - you might want better query extraction
                            tool_params = {"query": search_query, "limit": 5}
                        else:
                            # Try to parse parameters (basic JSON parsing)
                            try:
                                tool_params = json.loads(params_str)
                            except:
                                tool_params = {"query": user_query}
                        
                        # Call the MCP tool
                        tool_result = await self.mcp_manager.call_tool(mcp_server, tool_name, tool_params)
                        
                        # Generate final response with tool results
                        final_prompt = f"""Based on the following search results, please provide a comprehensive answer to the user's question.

User question: {user_query}

Search results:
{json.dumps(tool_result, indent=2)}

Please provide a natural, helpful response that incorporates the relevant information from the search results."""
                        
                        final_response = await provider.generate_text(
                            prompt=final_prompt,
                            model=model_id,
                            system=system_prompt,
                            max_tokens=2000
                        )
                        
                        return {
                            "provider": provider_name,
                            "model": model_id,
                            "mcp_server": mcp_server,
                            "response": final_response,
                            "tool_used": tool_name,
                            "tool_params": tool_params,
                            "tool_results": tool_result,
                            "raw_response": response
                        }
                    
                except Exception as e:
                    self.logger.error(f"Error processing tool usage: {e}")
                    # Return the original response if tool usage fails
                    pass
            
            return {
                "provider": provider_name,
                "model": model_id,
                "mcp_server": mcp_server,
                "response": response,
                "tool_used": None
            }
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources."""
        await self.mcp_manager.cleanup()


# Example usage and testing
async def main():
    """Main function for testing the orchestrator."""
    orchestrator = MCPOrchestrator()
    
    try:
        # Get available models
        models = orchestrator.get_available_models()
        print("Available models:")
        for provider, provider_models in models.items():
            print(f"\n{provider.upper()}:")
            for model in provider_models:
                print(f"  - {model['id']}: {model['name']}")
        
        # Get available MCP servers
        servers = orchestrator.mcp_manager.get_available_servers()
        print("\nAvailable MCP servers:")
        for server_id, server_info in servers.items():
            print(f"  - {server_id}: {server_info['name']} ({'installed' if server_info['installed'] else 'not installed'})")
        
        # Example query with web search
        if models and "anthropic" in models:
            print("\nTesting query with web search...")
            result = await orchestrator.execute_query(
                provider_name="anthropic",
                model_id="claude-3-5-sonnet-20241022",
                mcp_server="web-search",
                user_query="What are the latest developments in AI in 2025?",
                system_prompt="You are a helpful AI assistant that can search the web for current information."
            )
            
            print(f"\nResult:")
            print(json.dumps(result, indent=2))
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())