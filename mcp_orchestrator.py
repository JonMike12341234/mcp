#!/usr/bin/env python3
"""
FIXED MCP Integration - Proper LLM Function Calling Integration
This fixes the core issue where LLMs weren't aware of MCP server tools
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Import the existing classes
from mcp_orchestrator import RealMCPServerManager, EnhancedMCPClient
from providers.openai_provider import OpenAIProvider
from providers.gemini_provider import GeminiProvider  
from providers.anthropic_provider import AnthropicProvider
from utils.config import Config
from utils.logger import setup_logger


class MCPToolConverter:
    """Converts MCP tools to provider-specific function calling formats."""
    
    @staticmethod
    def mcp_to_openai_tools(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format."""
        openai_tools = []
        
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", "unknown_tool"),
                    "description": tool.get("description", "No description available"),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    @staticmethod
    def mcp_to_anthropic_tools(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic function calling format."""
        anthropic_tools = []
        
        for tool in mcp_tools:
            anthropic_tool = {
                "name": tool.get("name", "unknown_tool"),
                "description": tool.get("description", "No description available"),
                "input_schema": tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
            anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    @staticmethod
    def mcp_to_gemini_tools(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Gemini function calling format."""
        gemini_tools = []
        
        for tool in mcp_tools:
            gemini_tool = {
                "function_declarations": [{
                    "name": tool.get("name", "unknown_tool"),
                    "description": tool.get("description", "No description available"),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }]
            }
            gemini_tools.append(gemini_tool)
        
        return gemini_tools


class FixedMCPOrchestrator:
    """FIXED MCP Orchestrator with proper LLM function calling integration."""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger("fixed_mcp_orchestrator")
        
        # Initialize LLM providers
        self.providers = {
            'openai': OpenAIProvider(self.config.get_openai_config()),
            'gemini': GeminiProvider(self.config.get_gemini_config()),
            'anthropic': AnthropicProvider(self.config.get_anthropic_config())
        }
        
        # Initialize MCP server manager
        self.mcp_manager = RealMCPServerManager()
        
        # Initialize tool converter
        self.tool_converter = MCPToolConverter()
        
        self.logger.info("FIXED MCP Orchestrator initialized with proper LLM function calling")
    
    def get_available_models(self) -> Dict[str, List[Dict[str, str]]]:
        """Get available models from all providers."""
        models = {}
        
        for provider_name, provider in self.providers.items():
            if provider.is_available():
                try:
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
        """FIXED: Execute query with proper MCP tool integration."""
        try:
            # Validate provider
            if provider_name not in self.providers or not self.providers[provider_name].is_available():
                return {"error": f"Provider {provider_name} not available"}
            
            provider = self.providers[provider_name]
            
            # FIXED: Proper MCP tool integration
            if mcp_server:
                self.logger.info(f"FIXED: Connecting to MCP server and advertising tools to LLM: {mcp_server}")
                
                # Step 1: Connect to MCP server
                client = await self.mcp_manager.start_mcp_server(mcp_server)
                
                if client:
                    # Step 2: Get available tools from MCP server
                    mcp_tools = await self.mcp_manager.get_available_tools(mcp_server)
                    self.logger.info(f"FIXED: Got {len(mcp_tools)} tools from MCP server: {[t.get('name') for t in mcp_tools]}")
                    
                    if mcp_tools:
                        # Step 3: Convert MCP tools to provider-specific format
                        if provider_name == "openai":
                            provider_tools = self.tool_converter.mcp_to_openai_tools(mcp_tools)
                            response_data = await self._execute_openai_with_tools(
                                provider, model_id, user_query, system_prompt, provider_tools, mcp_server
                            )
                        elif provider_name == "anthropic":
                            provider_tools = self.tool_converter.mcp_to_anthropic_tools(mcp_tools)
                            response_data = await self._execute_anthropic_with_tools(
                                provider, model_id, user_query, system_prompt, provider_tools, mcp_server
                            )
                        elif provider_name == "gemini":
                            provider_tools = self.tool_converter.mcp_to_gemini_tools(mcp_tools)
                            response_data = await self._execute_gemini_with_tools(
                                provider, model_id, user_query, system_prompt, provider_tools, mcp_server
                            )
                        else:
                            return {"error": f"Provider {provider_name} not supported for tool calling"}
                        
                        return response_data
                    else:
                        self.logger.warning(f"No tools available from MCP server {mcp_server}")
                else:
                    self.logger.error(f"Failed to connect to MCP server {mcp_server}")
            
            # FIXED: Regular query without MCP server (no change needed here)
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
                "fixed_implementation": True
            }
            
        except Exception as e:
            self.logger.error(f"FIXED: Error executing query: {e}")
            return {"error": str(e)}
    
    async def _execute_openai_with_tools(
        self, 
        provider: OpenAIProvider, 
        model_id: str, 
        user_query: str, 
        system_prompt: Optional[str],
        tools: List[Dict[str, Any]],
        mcp_server: str
    ) -> Dict[str, Any]:
        """FIXED: Execute OpenAI request with MCP tools properly advertised."""
        try:
            # Step 1: Make initial request with tools advertised
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_query})
            
            self.logger.info(f"FIXED: Advertising {len(tools)} tools to OpenAI model")
            
            # Use the OpenAI client directly for function calling
            response = await provider.client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # Let the LLM decide when to use tools
                max_tokens=2000
            )
            
            message = response.choices[0].message
            
            # Step 2: Check if LLM wants to call tools
            if message.tool_calls:
                self.logger.info(f"FIXED: LLM requested {len(message.tool_calls)} tool calls")
                
                # Add the assistant's message to conversation
                messages.append({
                    "role": "assistant", 
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                # Execute each tool call via MCP
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    self.logger.info(f"FIXED: Executing MCP tool {tool_name} with args: {arguments}")
                    
                    # Call the MCP server
                    tool_result = await self.mcp_manager.call_tool(mcp_server, tool_name, arguments)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                
                # Step 3: Get final response from LLM with tool results
                final_response = await provider.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=2000
                )
                
                final_content = final_response.choices[0].message.content
                
                return {
                    "provider": "openai",
                    "model": model_id,
                    "mcp_server": mcp_server,
                    "response": final_content,
                    "tool_used": message.tool_calls[0].function.name if message.tool_calls else None,
                    "tool_calls_made": len(message.tool_calls),
                    "tools_available": len(tools),
                    "fixed_implementation": True
                }
            else:
                # No tools called, return direct response
                return {
                    "provider": "openai",
                    "model": model_id,
                    "mcp_server": mcp_server,
                    "response": message.content,
                    "tool_used": None,
                    "tools_available": len(tools),
                    "fixed_implementation": True
                }
                
        except Exception as e:
            self.logger.error(f"FIXED: Error in OpenAI tool execution: {e}")
            return {"error": f"OpenAI tool execution failed: {str(e)}"}
    
    async def _execute_anthropic_with_tools(
        self, 
        provider: AnthropicProvider, 
        model_id: str, 
        user_query: str, 
        system_prompt: Optional[str],
        tools: List[Dict[str, Any]],
        mcp_server: str
    ) -> Dict[str, Any]:
        """FIXED: Execute Anthropic request with MCP tools properly advertised."""
        try:
            messages = [{"role": "user", "content": user_query}]
            
            self.logger.info(f"FIXED: Advertising {len(tools)} tools to Anthropic model")
            
            # Step 1: Make request with tools
            response = await provider.client.messages.create(
                model=model_id,
                max_tokens=2000,
                messages=messages,
                tools=tools,
                system=system_prompt
            )
            
            # Step 2: Check if Claude wants to use tools
            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_args = content_block.input
                        
                        self.logger.info(f"FIXED: Anthropic requested tool {tool_name} with args: {tool_args}")
                        
                        # Execute tool via MCP
                        tool_result = await self.mcp_manager.call_tool(mcp_server, tool_name, tool_args)
                        
                        # Add tool result and get final response
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user", 
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": json.dumps(tool_result)
                            }]
                        })
                        
                        # Get final response
                        final_response = await provider.client.messages.create(
                            model=model_id,
                            max_tokens=2000,
                            messages=messages,
                            tools=tools,
                            system=system_prompt
                        )
                        
                        final_text = final_response.content[0].text if final_response.content else "No response"
                        
                        return {
                            "provider": "anthropic",
                            "model": model_id,
                            "mcp_server": mcp_server,
                            "response": final_text,
                            "tool_used": tool_name,
                            "tools_available": len(tools),
                            "fixed_implementation": True
                        }
                
                # No tool use, return direct response
                response_text = response.content[0].text if response.content else "No response"
                return {
                    "provider": "anthropic",
                    "model": model_id,
                    "mcp_server": mcp_server,
                    "response": response_text,
                    "tool_used": None,
                    "tools_available": len(tools),
                    "fixed_implementation": True
                }
                
        except Exception as e:
            self.logger.error(f"FIXED: Error in Anthropic tool execution: {e}")
            return {"error": f"Anthropic tool execution failed: {str(e)}"}
    
    async def _execute_gemini_with_tools(
        self, 
        provider: GeminiProvider, 
        model_id: str, 
        user_query: str, 
        system_prompt: Optional[str],
        tools: List[Dict[str, Any]],
        mcp_server: str
    ) -> Dict[str, Any]:
        """FIXED: Execute Gemini request with MCP tools properly advertised."""
        try:
            self.logger.info(f"FIXED: Advertising {len(tools)} tools to Gemini model")
            
            # Initialize model with system instruction
            model_kwargs = {}
            if system_prompt:
                model_kwargs["system_instruction"] = system_prompt
            
            generation_model = provider.client.GenerativeModel(model_id, **model_kwargs)
            
            # Step 1: Make request with tools
            response = generation_model.generate_content(
                user_query,
                tools=tools
            )
            
            # Step 2: Check if Gemini wants to use tools
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            # Tool call detected
                            func_call = part.function_call
                            tool_name = func_call.name
                            tool_args = dict(func_call.args)
                            
                            self.logger.info(f"FIXED: Gemini requested tool {tool_name} with args: {tool_args}")
                            
                            # Execute tool via MCP
                            tool_result = await self.mcp_manager.call_tool(mcp_server, tool_name, tool_args)
                            
                            # Prepare function response
                            function_response = {
                                "name": tool_name,
                                "response": tool_result
                            }
                            
                            # Get final response with tool result
                            final_response = generation_model.generate_content(
                                [
                                    {"role": "user", "parts": [{"text": user_query}]},
                                    {"role": "model", "parts": response.candidates[0].content.parts},
                                    {"role": "function", "parts": [{"function_response": function_response}]}
                                ]
                            )
                            
                            return {
                                "provider": "gemini",
                                "model": model_id,
                                "mcp_server": mcp_server,
                                "response": final_response.text,
                                "tool_used": tool_name,
                                "tools_available": len(tools),
                                "fixed_implementation": True
                            }
            
            # No tool use, return direct response
            return {
                "provider": "gemini",
                "model": model_id,
                "mcp_server": mcp_server,
                "response": response.text,
                "tool_used": None,
                "tools_available": len(tools),
                "fixed_implementation": True
            }
                
        except Exception as e:
            self.logger.error(f"FIXED: Error in Gemini tool execution: {e}")
            return {"error": f"Gemini tool execution failed: {str(e)}"}
    
    async def cleanup(self):
        """Clean up resources."""
        await self.mcp_manager.cleanup()


# Example usage
async def test_fixed_mcp():
    """Test the fixed MCP integration."""
    orchestrator = FixedMCPOrchestrator()
    
    try:
        print("🔧 Testing FIXED MCP Integration")
        print("=" * 50)
        
        # Test web search with proper tool integration
        result = await orchestrator.execute_query(
            provider_name="openai",
            model_id="gpt-4o",
            mcp_server="brave-search",
            user_query="What are the latest developments in AI in 2025?",
            system_prompt="You are a helpful assistant with access to web search. Use tools when needed."
        )
        
        print("FIXED Integration Result:")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Tool Used: {result.get('tool_used', 'None')}")
        print(f"Tools Available: {result.get('tools_available', 0)}")
        print(f"Fixed Implementation: {result.get('fixed_implementation', False)}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(test_fixed_mcp())