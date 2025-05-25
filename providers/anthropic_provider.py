"""
Anthropic Claude Provider for Universal MCP Server
Supports Claude Opus 4, Sonnet 4, and other Anthropic models
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AsyncAnthropic = None

from mcp.types import Resource, Tool, TextContent
from .base_provider import BaseProvider

class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider implementation for the Universal MCP Server."""
    
    # Available models with their specifications
    MODELS = {
        # Claude 4 Series (Latest - May 2025)
        "claude-opus-4-20250514": {
            "context_length": 200000,
            "max_output_tokens": 32000,
            "input_cost_per_1m": 15.0,
            "output_cost_per_1m": 75.0,
            "description": "Most capable Claude model with highest level of intelligence",
            "supports_reasoning": True,
            "supports_tool_use": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "supports_extended_thinking": True,
            "priority_tier": True,
            "training_cutoff": "Mar 2025"
        },
        "claude-sonnet-4-20250514": {
            "context_length": 200000,
            "max_output_tokens": 64000,
            "input_cost_per_1m": 3.0,
            "output_cost_per_1m": 15.0,
            "description": "High-performance model with balanced intelligence and speed",
            "supports_reasoning": True,
            "supports_tool_use": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "supports_extended_thinking": True,
            "priority_tier": True,
            "training_cutoff": "Mar 2025"
        },
        
        # Claude 3.7 Series (February 2025)
        "claude-3-7-sonnet-20250219": {
            "context_length": 200000,
            "max_output_tokens": 64000,
            "input_cost_per_1m": 3.0,
            "output_cost_per_1m": 15.0,
            "description": "High-performance model with toggleable extended thinking",
            "supports_reasoning": True,
            "supports_tool_use": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "supports_extended_thinking": True,
            "priority_tier": True,
            "training_cutoff": "Nov 2024"
        },
        
        # Claude 3.5 Series (Established)
        "claude-3-5-sonnet-20241022": {
            "context_length": 200000,
            "max_output_tokens": 64000,
            "input_cost_per_1m": 3.0,
            "output_cost_per_1m": 15.0,
            "description": "High intelligence and balanced performance model",
            "supports_reasoning": False,
            "supports_tool_use": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "supports_extended_thinking": False,
            "priority_tier": True,
            "training_cutoff": "Mar 2025"
        },
        "claude-3-5-haiku-20241022": {
            "context_length": 200000,
            "max_output_tokens": 8192,
            "input_cost_per_1m": 0.8,
            "output_cost_per_1m": 4.0,
            "description": "Fastest model with intelligence at blazing speed",
            "supports_reasoning": False,
            "supports_tool_use": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "supports_extended_thinking": False,
            "priority_tier": True,
            "training_cutoff": "July 2024"
        },
        
        # Claude 3 Series (Legacy)
        "claude-3-opus-20240229": {
            "context_length": 200000,
            "max_output_tokens": 8192,
            "input_cost_per_1m": 15.0,
            "output_cost_per_1m": 75.0,
            "description": "Previous intelligent model (legacy)",
            "supports_reasoning": False,
            "supports_tool_use": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "supports_extended_thinking": False,
            "priority_tier": True,
            "training_cutoff": "Apr 2024"
        },
        "claude-3-haiku-20240307": {
            "context_length": 200000,
            "max_output_tokens": 8192,
            "input_cost_per_1m": 0.25,
            "output_cost_per_1m": 1.25,
            "description": "Legacy fast model",
            "supports_reasoning": False,
            "supports_tool_use": True,
            "supports_vision": True,
            "supports_multimodal": True,
            "supports_extended_thinking": False,
            "priority_tier": True,
            "training_cutoff": "Apr 2024"
        },
        
        # Aliases for latest versions
        "claude-opus-4-0": {
            "alias_for": "claude-opus-4-20250514"
        },
        "claude-sonnet-4-0": {
            "alias_for": "claude-sonnet-4-20250514"  
        },
        "claude-3-7-sonnet-latest": {
            "alias_for": "claude-3-7-sonnet-20250219"
        },
        "claude-3-5-sonnet-latest": {
            "alias_for": "claude-3-5-sonnet-20241022"
        },
        "claude-3-5-haiku-latest": {
            "alias_for": "claude-3-5-haiku-20241022"
        },
        "claude-3-opus-latest": {
            "alias_for": "claude-3-opus-20240229"
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("anthropic", config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.default_model = config.get("default_model", "claude-3-5-sonnet-20241022")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60)
        
        # Statistics tracking
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.last_request_time = None
        
        if not HAS_ANTHROPIC:
            self.logger.warning("Anthropic library not installed")
            self._available = False
            self.client = None
        elif not self.api_key:
            self.logger.warning("Anthropic API key not provided")
            self._available = False
            self.client = None
        else:
            try:
                self.client = AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=self.max_retries,
                    timeout=self.timeout
                )
                self._available = True
                self.logger.info("Anthropic provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Anthropic client: {e}")
                self._available = False
                self.client = None
    
    def _resolve_model_alias(self, model: str) -> str:
        """Resolve model alias to actual model name."""
        if model in self.MODELS and "alias_for" in self.MODELS[model]:
            return self.MODELS[model]["alias_for"]
        return model
    
    def _get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information, resolving aliases."""
        resolved_model = self._resolve_model_alias(model)
        return self.MODELS.get(resolved_model, {})
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._available and self.client is not None
    
    async def get_status(self) -> Dict[str, Any]:
        """Get provider status information."""
        if not self.is_available():
            return {
                "available": False,
                "error": "API key not configured or Anthropic library not installed"
            }
        
        try:
            # Resolve the default model alias before using it
            resolved_model = self._resolve_model_alias(self.default_model)
            
            # Perform a simple request to check connectivity and validity
            response = await self.client.messages.create(
                model=resolved_model,
                max_tokens=1,
                messages=[
                    {"role": "user", "content": "Hello"}
                ]
            )
            return {
                "available": True,
                "api_status": "connected",
                "default_model": self.default_model,
                "resolved_model": resolved_model,
                "request_count": self.request_count,
                "total_tokens_used": self.total_tokens_used,
                "total_cost_usd": round(self.total_cost, 4),
                "last_request": self.last_request_time,
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            error_message = str(e)
            
            # Handle specific error types more gracefully
            if "credit balance" in error_message.lower():
                return {
                    "available": False,
                    "error": "Insufficient credits - Please add credits to your Anthropic account",
                    "error_type": "billing",
                    "last_checked": datetime.now().isoformat()
                }
            elif "unauthorized" in error_message.lower() or "invalid" in error_message.lower():
                return {
                    "available": False,
                    "error": "Invalid API key - Please check your Anthropic API key",
                    "error_type": "authentication",
                    "last_checked": datetime.now().isoformat()
                }
            elif "not_found_error" in error_message.lower() and "model:" in error_message.lower():
                return {
                    "available": False,
                    "error": f"Model not found - Using: {self.default_model} (resolved to: {self._resolve_model_alias(self.default_model)})",
                    "error_type": "model_error",
                    "last_checked": datetime.now().isoformat()
                }
            else:
                return {
                    "available": False,
                    "error": error_message,
                    "error_type": "unknown",
                    "last_checked": datetime.now().isoformat()
                }    
    async def list_models(self) -> Dict[str, Any]:
        """List available models with their specifications."""
        if not self.is_available():
            return {"error": "Provider not available"}
        
        # Filter out aliases and return actual models with their specs
        actual_models = {}
        aliases = {}
        
        for model_id, info in self.MODELS.items():
            if "alias_for" in info:
                aliases[model_id] = info["alias_for"]
            else:
                actual_models[model_id] = info
        
        return {
            "provider": "anthropic",
            "models": actual_models,
            "aliases": aliases,
            "default_model": self.default_model,
            "total_count": len(actual_models),
            "alias_count": len(aliases)
        }
    
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using Claude models."""
        if not self.is_available():
            raise ValueError("Anthropic provider not available")
        
        model = model or self.default_model
        # Resolve model alias to actual model name
        resolved_model = self._resolve_model_alias(model)
        max_tokens = max_tokens or 1000
        temperature = temperature or 0.7
        
        try:
            # Prepare message parameters
            message_params = {
                "model": resolved_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add system prompt if provided
            if system:
                message_params["system"] = system
            
            # Add any additional parameters
            message_params.update(kwargs)
            
            # Make the API call
            response = await self.client.messages.create(**message_params)
            
            # Update statistics
            self.request_count += 1
            self.last_request_time = datetime.now().isoformat()
            
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                self.total_tokens_used += (input_tokens + output_tokens)
                
                # Calculate cost using resolved model
                model_info = self._get_model_info(model)
                if model_info:
                    input_cost = (input_tokens / 1_000_000) * model_info["input_cost_per_1m"]
                    output_cost = (output_tokens / 1_000_000) * model_info["output_cost_per_1m"]
                    self.total_cost += (input_cost + output_cost)
            
            # Extract text content from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"Text generation error: {e}")
            raise
    
    async def get_available_tools(self) -> List[Tool]:
        """Get provider-specific tools."""
        if not self.is_available():
            return []
        
        return [
            Tool(
                name="chat_with_system",
                description="Chat with Claude using a system prompt and user message",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "system": {
                            "type": "string",
                            "description": "System prompt to set Claude's behavior"
                        },
                        "message": {
                            "type": "string",
                            "description": "User message to send to Claude"
                        },
                        "model": {
                            "type": "string",
                            "description": "Claude model to use",
                            "enum": [
                                "claude-opus-4-20250514", "claude-sonnet-4-20250514", 
                                "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", 
                                "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
                                "claude-opus-4-0", "claude-sonnet-4-0", "claude-3-7-sonnet-latest",
                                "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"
                            ]
                        },
                        "max_tokens": {
                            "type": "integer",
                            "default": 1000,
                            "description": "Maximum tokens to generate"
                        },
                        "temperature": {
                            "type": "number",
                            "default": 0.7,
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Sampling temperature"
                        }
                    },
                    "required": ["message"]
                }
            ),
            Tool(
                name="multi_turn_conversation",
                description="Have a multi-turn conversation with Claude",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": ["user", "assistant"]
                                    },
                                    "content": {
                                        "type": "string"
                                    }
                                },
                                "required": ["role", "content"]
                            },
                            "description": "List of conversation messages"
                        },
                        "system": {
                            "type": "string",
                            "description": "System prompt (optional)"
                        },
                        "model": {
                            "type": "string",
                            "description": "Claude model to use",
                            "enum": [
                                "claude-opus-4-20250514", "claude-sonnet-4-20250514", 
                                "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", 
                                "claude-3-5-haiku-20241022", "claude-3-opus-20240229"
                            ]
                        }
                    },
                    "required": ["messages"]
                }
            ),
            Tool(
                name="analyze_with_reasoning",
                description="Use Claude's reasoning capabilities for complex analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The analysis task to perform"
                        },
                        "data": {
                            "type": "string",
                            "description": "Data or context for analysis"
                        },
                        "reasoning_model": {
                            "type": "string",
                            "enum": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"],
                            "description": "Model with reasoning capabilities"
                        }
                    },
                    "required": ["task", "data"]
                }
            )
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a provider-specific tool."""
        if not self.is_available():
            raise ValueError("Provider not available")
        
        try:
            if tool_name == "chat_with_system":
                return await self._handle_chat_with_system(arguments)
            elif tool_name == "multi_turn_conversation":
                return await self._handle_multi_turn_conversation(arguments)
            elif tool_name == "analyze_with_reasoning":
                return await self._handle_analyze_with_reasoning(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {tool_name}")]
        except Exception as e:
            self.logger.error(f"Error calling Anthropic tool {tool_name}: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _handle_chat_with_system(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle chat with system prompt."""
        system = arguments.get("system")
        message = arguments.get("message")
        model = arguments.get("model", self.default_model)
        max_tokens = arguments.get("max_tokens", 1000)
        temperature = arguments.get("temperature", 0.7)
        
        try:
            response = await self.generate_text(
                prompt=message,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system
            )
            return [TextContent(type="text", text=response)]
        except Exception as e:
            raise ValueError(f"Error in chat with system: {str(e)}")
    
    async def _handle_multi_turn_conversation(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle multi-turn conversation."""
        messages = arguments.get("messages", [])
        system = arguments.get("system")
        model = arguments.get("model", self.default_model)
        
        try:
            message_params = {
                "model": model,
                "max_tokens": 2000,
                "messages": messages
            }
            
            if system:
                message_params["system"] = system
            
            response = await self.client.messages.create(**message_params)
            
            if response.content and len(response.content) > 0:
                return [TextContent(type="text", text=response.content[0].text)]
            else:
                return [TextContent(type="text", text="No response generated")]
                
        except Exception as e:
            raise ValueError(f"Error in multi-turn conversation: {str(e)}")
    
    async def _handle_analyze_with_reasoning(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle analysis with reasoning capabilities."""
        task = arguments.get("task")
        data = arguments.get("data")
        reasoning_model = arguments.get("reasoning_model", "claude-sonnet-4-20250514")
        
        # Verify the model supports reasoning
        model_info = self._get_model_info(reasoning_model)
        if model_info and model_info.get("supports_reasoning"):
            system_prompt = """You are an expert analyst with advanced reasoning capabilities. 
            Break down complex problems step by step, show your reasoning process, 
            and provide thorough analysis with supporting evidence."""
            
            analysis_prompt = f"""Task: {task}

Data/Context:
{data}

Please provide a detailed analysis following these steps:
1. Understanding the problem/question
2. Identifying key factors and relationships
3. Step-by-step reasoning process
4. Conclusions and recommendations
5. Confidence level and limitations

Be thorough and show your reasoning at each step."""
            
            try:
                response = await self.generate_text(
                    prompt=analysis_prompt,
                    model=reasoning_model,
                    max_tokens=4000,
                    system=system_prompt
                )
                return [TextContent(type="text", text=response)]
            except Exception as e:
                raise ValueError(f"Error in reasoning analysis: {str(e)}")
        else:
            return [TextContent(type="text", text=f"Model {reasoning_model} does not support advanced reasoning")]
    
    async def get_available_resources(self) -> List[Resource]:
        """Get provider-specific resources."""
        if not self.is_available():
            return []
        
        return [
            Resource(
                uri="anthropic://models",
                description="Information about available Anthropic models",
                mimeType="application/json"
            ),
            Resource(
                uri="anthropic://usage",
                description="Current usage statistics for Anthropic provider",
                mimeType="application/json"
            ),
            Resource(
                uri="anthropic://capabilities",
                description="Capabilities and features of Anthropic models",
                mimeType="application/json"
            ),
            Resource(
                uri="anthropic://pricing",
                description="Pricing information for Anthropic models",
                mimeType="application/json"
            ),
            Resource(
                uri="anthropic://limits",
                description="Rate limits and token limits for Anthropic models",
                mimeType="application/json"
            )
        ]
    
    async def get_resource(self, uri: str) -> str:
        """Get a provider-specific resource."""
        if not self.is_available():
            raise ValueError("Provider not available")
        
        try:
            if uri == "models":
                models_info = await self.list_models()
                return json.dumps(models_info, indent=2)
            
            elif uri == "usage":
                usage_stats = await self.get_usage_stats()
                return json.dumps(usage_stats, indent=2)
            
            elif uri == "capabilities":
                capabilities = {
                    "provider": "anthropic",
                    "features": {
                        "text_generation": True,
                        "conversation": True,
                        "system_prompts": True,
                        "reasoning": True,
                        "tool_use": True,
                        "vision": True,
                        "function_calling": True,
                        "streaming": True
                    },
                    "model_capabilities": {
                        model: {
                            "reasoning": info.get("supports_reasoning", False),
                            "tool_use": info.get("supports_tool_use", False),
                            "vision": info.get("supports_vision", False),
                            "context_length": info.get("context_length"),
                            "max_output": info.get("max_output_tokens")
                        }
                        for model, info in self.MODELS.items()
                    }
                }
                return json.dumps(capabilities, indent=2)
            
            elif uri == "pricing":
                pricing_info = {
                    "provider": "anthropic",
                    "currency": "USD",
                    "pricing_model": "per_million_tokens",
                    "models": {
                        model: {
                            "input_cost_per_1m_tokens": info["input_cost_per_1m"],
                            "output_cost_per_1m_tokens": info["output_cost_per_1m"],
                            "description": info["description"]
                        }
                        for model, info in self.MODELS.items()
                    }
                }
                return json.dumps(pricing_info, indent=2)
            
            elif uri == "limits":
                limits_info = {
                    "provider": "anthropic",
                    "rate_limits": {
                        "requests_per_minute": 60,
                        "requests_per_day": 1000,
                        "tokens_per_minute": 40000,
                        "concurrent_requests": 5
                    },
                    "model_limits": {
                        model: {
                            "context_length": info["context_length"],
                            "max_output_tokens": info["max_output_tokens"]
                        }
                        for model, info in self.MODELS.items()
                    }
                }
                return json.dumps(limits_info, indent=2)
            
            else:
                return json.dumps({"error": f"Unknown resource: {uri}"})
                
        except Exception as e:
            self.logger.error(f"Error getting Anthropic resource {uri}: {e}")
            return json.dumps({"error": str(e)})
    
    async def estimate_cost(self, prompt: str, max_tokens: int = 1000, model: Optional[str] = None) -> Dict[str, Any]:
        """Estimate the cost of a request."""
        model = model or self.default_model
        model_info = self._get_model_info(model)
        
        if not model_info:
            return {"error": f"Unknown model: {model}"}
            
        # Simple token estimation (roughly 4 characters per token for English)
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = max_tokens
        
        input_cost = (estimated_input_tokens / 1_000_000) * model_info["input_cost_per_1m"]
        output_cost = (estimated_output_tokens / 1_000_000) * model_info["output_cost_per_1m"]
        total_cost = input_cost + output_cost
        
        return {
            "provider": "anthropic",
            "model": model,
            "resolved_model": self._resolve_model_alias(model),
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "currency": "USD",
            "pricing_model": "per_million_tokens",
            "supports_extended_thinking": model_info.get("supports_extended_thinking", False),
            "context_length": model_info.get("context_length"),
            "max_output_tokens": model_info.get("max_output_tokens")
        }
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this provider."""
        return {
            "provider": "anthropic",
            "default_model": self.default_model,
            "status": "active" if self.is_available() else "inactive",
            "statistics": {
                "total_requests": self.request_count,
                "total_tokens_used": self.total_tokens_used,
                "total_cost_usd": round(self.total_cost, 4),
                "last_request": self.last_request_time,
                "average_cost_per_request": round(self.total_cost / max(self.request_count, 1), 4)
            },
            "configuration": {
                "base_url": self.base_url,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "models_available": len(self.MODELS)
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.last_request_time = None
        self.logger.info("Anthropic provider statistics reset")