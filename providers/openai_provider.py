"""
OpenAI Provider for Universal MCP Server
Supports GPT-4o, GPT-4 Turbo, o1-series, and other OpenAI models
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import base64

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = None

from mcp.types import Resource, Tool, TextContent
from .base_provider import BaseProvider

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation for the Universal MCP Server."""
    
    # Available models with their specifications
    MODELS = {
        # GPT-4 Series (Current - May 2025)
        "gpt-4o": {
            "context_length": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_1m": 5.0,
            "output_cost_per_1m": 15.0,
            "description": "GPT-4 Omni - multimodal flagship model",
            "supports_function_calling": True,
            "supports_vision": True
        },
        "gpt-4o-mini": {
            "context_length": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_1m": 0.15,
            "output_cost_per_1m": 0.6,
            "description": "Efficient small model with strong performance",
            "supports_function_calling": True,
            "supports_vision": True
        },
        "gpt-4-turbo": {
            "context_length": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_1m": 10.0,
            "output_cost_per_1m": 30.0,
            "description": "Previous generation GPT-4 Turbo",
            "supports_function_calling": True,
            "supports_vision": True
        },

        # o1 Series (Reasoning Models)
        "o1-preview": {
            "context_length": 128000,
            "max_output_tokens": 32768,
            "input_cost_per_1m": 15.0,
            "output_cost_per_1m": 60.0,
            "description": "Reasoning model optimized for complex problem solving",
            "supports_function_calling": False,
            "supports_vision": True
        },
        "o1-mini": {
            "context_length": 128000,
            "max_output_tokens": 65536,
            "input_cost_per_1m": 3.0,
            "output_cost_per_1m": 12.0,
            "description": "Efficient reasoning model for coding and STEM",
            "supports_function_calling": False,
            "supports_vision": False
        },
        
        # GPT-3.5 Series
        "gpt-3.5-turbo": {
            "context_length": 16385,
            "max_output_tokens": 4096,
            "input_cost_per_1m": 0.5,
            "output_cost_per_1m": 1.5,
            "description": "Fast and cost-effective model for simple tasks",
            "supports_function_calling": True,
            "supports_vision": False
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.organization = config.get("organization") if config.get("organization") else None
        self.default_model = config.get("default_model", "gpt-4o")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60)
        
        # Statistics tracking
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.last_request_time = None
        
        if not HAS_OPENAI:
            self.logger.warning("OpenAI library not installed")
            self._available = False
            self.client = None
        elif not self.api_key:
            self.logger.warning("OpenAI API key not provided")
            self._available = False
            self.client = None
        else:
            try:
                # Only pass organization if it's actually provided and not empty
                client_kwargs = {
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                    "max_retries": self.max_retries,
                    "timeout": self.timeout
                }
                
                # Only add organization if it's a valid non-empty string
                if self.organization and self.organization.strip() and self.organization.lower() not in ['none', 'null', 'na']:
                    client_kwargs["organization"] = self.organization
                
                self.client = AsyncOpenAI(**client_kwargs)
                self._available = True
                self.logger.info("OpenAI provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self._available = False
                self.client = None
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._available and self.client is not None
    
    async def get_status(self) -> Dict[str, Any]:
        """Get provider status information."""
        if not self.is_available():
            return {
                "available": False,
                "error": "API key not configured or OpenAI library not installed"
            }
        
        try:
            # Test API connection by listing models
            models = await self.client.models.list()
            return {
                "available": True,
                "api_status": "connected",
                "models_available": len(models.data),
                "default_model": self.default_model,
                "organization": self.organization,
                "request_count": self.request_count,
                "total_tokens_used": self.total_tokens_used,
                "total_cost_usd": round(self.total_cost, 4),
                "last_request": self.last_request_time,
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            error_message = str(e)
            
            # Handle specific error types more gracefully
            if "organization" in error_message.lower():
                return {
                    "available": False,
                    "error": "Invalid organization ID - Remove OPENAI_ORGANIZATION from .env if you don't have one",
                    "error_type": "configuration",
                    "last_checked": datetime.now().isoformat()
                }
            elif "incorrect api key" in error_message.lower() or "invalid" in error_message.lower():
                return {
                    "available": False,
                    "error": "Invalid API key - Please check your OpenAI API key",
                    "error_type": "authentication",
                    "last_checked": datetime.now().isoformat()
                }
            elif "quota" in error_message.lower() or "billing" in error_message.lower():
                return {
                    "available": False,
                    "error": "Quota exceeded - Please add billing information or credits",
                    "error_type": "billing",
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
        
        try:
            # Get models from API
            api_models = await self.client.models.list()
            api_model_ids = [model.id for model in api_models.data]
            
            # Filter our known models by what's actually available
            available_models = {}
            for model_id, specs in self.MODELS.items():
                if model_id in api_model_ids:
                    available_models[model_id] = specs
            
            return {
                "provider": "openai",
                "models": available_models,
                "default_model": self.default_model,
                "total_count": len(available_models),
                "api_models_found": len(api_model_ids)
            }
        except Exception as e:
            self.logger.error(f"Error listing OpenAI models: {e}")
            return {"error": str(e)}
    
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI models."""
        if not self.is_available():
            raise ValueError("OpenAI provider not available")
        
        model = model or self.default_model
        max_tokens = max_tokens or 1000
        temperature = temperature or 0.7
        
        try:
            # Prepare messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            response = await self.client.chat.completions.create(**params)
            
            # Update statistics
            self.request_count += 1
            self.last_request_time = datetime.now().isoformat()
            
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self.total_tokens_used += (input_tokens + output_tokens)
                
                # Calculate cost
                if model in self.MODELS:
                    model_info = self.MODELS[model]
                    input_cost = (input_tokens / 1_000_000) * model_info["input_cost_per_1m"]
                    output_cost = (output_tokens / 1_000_000) * model_info["output_cost_per_1m"]
                    self.total_cost += (input_cost + output_cost)
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating text with OpenAI: {e}")
            raise
    
    async def get_available_tools(self) -> List[Tool]:
        """Get provider-specific tools."""
        if not self.is_available():
            return []
        
        return [
            Tool(
                name="chat_completion",
                description="Create a chat completion with OpenAI models",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                    "content": {"type": "string"}
                                },
                                "required": ["role", "content"]
                            },
                            "description": "List of chat messages"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use for completion",
                            "enum": list(self.MODELS.keys())
                        },
                        "temperature": {
                            "type": "number",
                            "default": 0.7,
                            "minimum": 0,
                            "maximum": 2,
                            "description": "Sampling temperature"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "default": 1000,
                            "description": "Maximum tokens to generate"
                        }
                    },
                    "required": ["messages"]
                }
            ),
            Tool(
                name="function_calling",
                description="Use OpenAI's function calling capabilities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"}
                                }
                            },
                            "description": "Conversation messages"
                        },
                        "functions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "parameters": {"type": "object"}
                                }
                            },
                            "description": "Functions available to the model"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use",
                            "enum": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
                        }
                    },
                    "required": ["messages", "functions"]
                }
            ),
            Tool(
                name="vision_analysis",
                description="Analyze images using OpenAI's vision capabilities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt about the image"
                        },
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to analyze"
                        },
                        "image_data": {
                            "type": "string",
                            "description": "Base64 encoded image data (alternative to URL)"
                        },
                        "model": {
                            "type": "string",
                            "description": "Vision-capable model to use",
                            "enum": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
                        },
                        "max_tokens": {
                            "type": "integer",
                            "default": 300,
                            "description": "Maximum tokens for analysis"
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="code_analysis",
                description="Analyze and explain code using OpenAI models",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to analyze"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language of the code"
                        },
                        "task": {
                            "type": "string",
                            "description": "What to do with the code",
                            "enum": ["explain", "debug", "optimize", "review", "convert"]
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use for analysis",
                            "enum": list(self.MODELS.keys())
                        }
                    },
                    "required": ["code", "task"]
                }
            )
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a provider-specific tool."""
        if not self.is_available():
            raise ValueError("Provider not available")
        
        try:
            if tool_name == "chat_completion":
                return await self._handle_chat_completion(arguments)
            elif tool_name == "function_calling":
                return await self._handle_function_calling(arguments)
            elif tool_name == "vision_analysis":
                return await self._handle_vision_analysis(arguments)
            elif tool_name == "code_analysis":
                return await self._handle_code_analysis(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {tool_name}")]
        except Exception as e:
            self.logger.error(f"Error calling OpenAI tool {tool_name}: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _handle_chat_completion(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle chat completion requests."""
        messages = arguments.get("messages", [])
        model = arguments.get("model", self.default_model)
        temperature = arguments.get("temperature", 0.7)
        max_tokens = arguments.get("max_tokens", 1000)
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return [TextContent(type="text", text=response.choices[0].message.content)]
        except Exception as e:
            raise ValueError(f"Error in chat completion: {str(e)}")
    
    async def _handle_function_calling(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle function calling requests."""
        messages = arguments.get("messages", [])
        functions = arguments.get("functions", [])
        model = arguments.get("model", self.default_model)
        
        # Verify model supports function calling
        if model in self.MODELS and not self.MODELS[model].get("supports_function_calling"):
            return [TextContent(type="text", text=f"Model {model} does not support function calling")]
        
        try:
            # Convert functions to OpenAI format
            tools = []
            for func in functions:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": func["parameters"]
                    }
                }
                tools.append(tool_def)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            result_text = ""
            
            if message.content:
                result_text += message.content
            
            if message.tool_calls:
                result_text += "\n\nFunction calls made:"
                for tool_call in message.tool_calls:
                    result_text += f"\n- {tool_call.function.name}({tool_call.function.arguments})"
            
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            raise ValueError(f"Error in function calling: {str(e)}")
    
    async def _handle_vision_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle vision analysis requests."""
        prompt = arguments.get("prompt")
        image_url = arguments.get("image_url")
        image_data = arguments.get("image_data")
        model = arguments.get("model", "gpt-4o")
        max_tokens = arguments.get("max_tokens", 300)
        
        # Verify model supports vision
        if model in self.MODELS and not self.MODELS[model].get("supports_vision"):
            return [TextContent(type="text", text=f"Model {model} does not support vision")]
        
        if not image_url and not image_data:
            return [TextContent(type="text", text="Either image_url or image_data must be provided")]
        
        try:
            # Prepare content
            content = [{"type": "text", "text": prompt}]
            
            if image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            elif image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens
            )
            
            return [TextContent(type="text", text=response.choices[0].message.content)]
            
        except Exception as e:
            raise ValueError(f"Error in vision analysis: {str(e)}")
    
    async def _handle_code_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle code analysis requests."""
        code = arguments.get("code")
        language = arguments.get("language", "unknown")
        task = arguments.get("task")
        model = arguments.get("model", self.default_model)
        
        try:
            # Create appropriate system prompt based on task
            task_prompts = {
                "explain": "You are a code expert. Explain the provided code clearly and concisely.",
                "debug": "You are a debugging expert. Identify potential bugs and issues in the code.",
                "optimize": "You are a performance expert. Suggest optimizations for the provided code.",
                "review": "You are a code reviewer. Provide a comprehensive code review with suggestions.",
                "convert": "You are a code conversion expert. Help convert code between languages or formats."
            }
            
            system_prompt = task_prompts.get(task, "You are a helpful coding assistant.")
            
            user_prompt = f"""Task: {task.title()} the following {language} code:

```{language}
{code}
```

Please provide a detailed analysis."""
            
            response = await self.generate_text(
                prompt=user_prompt,
                model=model,
                system=system_prompt,
                max_tokens=2000
            )
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            raise ValueError(f"Error in code analysis: {str(e)}")
    
    async def get_available_resources(self) -> List[Resource]:
        """Get provider-specific resources."""
        if not self.is_available():
            return []
        
        return [
            Resource(
                uri="openai://models",
                description="Information about available OpenAI models",
                mimeType="application/json"
            ),
            Resource(
                uri="openai://usage",
                description="Current usage statistics for OpenAI provider",
                mimeType="application/json"
            ),
            Resource(
                uri="openai://capabilities",
                description="Capabilities and features of OpenAI models",
                mimeType="application/json"
            ),
            Resource(
                uri="openai://pricing",
                description="Pricing information for OpenAI models",
                mimeType="application/json"
            ),
            Resource(
                uri="openai://limits",
                description="Rate limits and usage limits for OpenAI models",
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
                    "provider": "openai",
                    "features": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision_analysis": True,
                        "code_analysis": True,
                        "streaming": True,
                        "fine_tuning": True
                    },
                    "model_capabilities": {
                        model: {
                            "function_calling": info.get("supports_function_calling", False),
                            "vision": info.get("supports_vision", False),
                            "context_length": info.get("context_length"),
                            "max_output": info.get("max_output_tokens")
                        }
                        for model, info in self.MODELS.items()
                    },
                    "supported_formats": ["text", "image", "function_calls"],
                    "reasoning_models": ["o1-preview", "o1-mini"]
                }
                return json.dumps(capabilities, indent=2)
            
            elif uri == "pricing":
                pricing_info = {
                    "provider": "openai",
                    "currency": "USD",
                    "pricing_model": "per_million_tokens",
                    "models": {
                        model: {
                            "input_cost_per_1m_tokens": info["input_cost_per_1m"],
                            "output_cost_per_1m_tokens": info["output_cost_per_1m"],
                            "description": info["description"],
                            "context_length": info["context_length"]
                        }
                        for model, info in self.MODELS.items()
                    },
                    "billing_info": {
                        "pay_per_use": True,
                        "monthly_billing": True,
                        "prepaid_credits": True
                    }
                }
                return json.dumps(pricing_info, indent=2)
            
            elif uri == "limits":
                limits_info = {
                    "provider": "openai",
                    "rate_limits": {
                        "tier_1": {
                            "requests_per_minute": 500,
                            "tokens_per_minute": 30000,
                            "requests_per_day": 10000
                        },
                        "tier_2": {
                            "requests_per_minute": 5000,
                            "tokens_per_minute": 450000,
                            "requests_per_day": 10000
                        }
                    },
                    "model_limits": {
                        model: {
                            "context_length": info["context_length"],
                            "max_output_tokens": info["max_output_tokens"]
                        }
                        for model, info in self.MODELS.items()
                    },
                    "usage_policies": {
                        "content_policy": True,
                        "rate_limiting": True,
                        "fair_use": True
                    }
                }
                return json.dumps(limits_info, indent=2)
            
            else:
                return json.dumps({"error": f"Unknown resource: {uri}"})
                
        except Exception as e:
            self.logger.error(f"Error getting OpenAI resource {uri}: {e}")
            return json.dumps({"error": str(e)})
    
    async def estimate_cost(self, prompt: str, max_tokens: int = 1000, model: Optional[str] = None) -> Dict[str, Any]:
        """Estimate the cost of a request."""
        model = model or self.default_model
        
        if model not in self.MODELS:
            return {"error": f"Unknown model: {model}"}
        
        # Simple token estimation (roughly 4 characters per token for English)
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = max_tokens
        
        model_info = self.MODELS[model]
        input_cost = (estimated_input_tokens / 1_000_000) * model_info["input_cost_per_1m"]
        output_cost = (estimated_output_tokens / 1_000_000) * model_info["output_cost_per_1m"]
        total_cost = input_cost + output_cost
        
        return {
            "provider": "openai",
            "model": model,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "currency": "USD",
            "pricing_model": "per_million_tokens",
            "supports_vision": model_info.get("supports_vision", False),
            "supports_functions": model_info.get("supports_function_calling", False)
        }
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this provider."""
        return {
            "provider": "openai",
            "default_model": self.default_model,
            "organization": self.organization,
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
            "capabilities_summary": {
                "text_generation": True,
                "vision_analysis": True,
                "function_calling": True,
                "code_analysis": True,
                "reasoning_models": True
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.last_request_time = None
        self.logger.info("OpenAI provider statistics reset")