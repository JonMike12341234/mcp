"""
Google Gemini Provider for Universal MCP Server
Supports Gemini 2.0/2.5 Pro, Flash, and other Google AI models
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import base64
import io

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

from mcp.types import Resource, Tool, TextContent
from .base_provider import BaseProvider

class GeminiProvider(BaseProvider):
    """Google Gemini provider implementation for the Universal MCP Server."""
    
    # Available models with their specifications
    MODELS = {
        # Gemini 2.5 Series (Latest - March 2025)
        "gemini-2.5-pro": {
            "context_length": 1000000,  # 2M coming soon
            "max_output_tokens": 32000,
            "input_cost_per_1m": 3.5,
            "output_cost_per_1m": 10.5,
            "description": "Most advanced Gemini model with enhanced reasoning and Deep Think mode",
            "multimodal": True,
            "supports_function_calling": True,
            "supports_code_execution": True,
            "supports_search": True
        },
        "gemini-2.5-flash": {
            "context_length": 1000000,
            "max_output_tokens": 16000,
            "input_cost_per_1m": 0.5,
            "output_cost_per_1m": 1.5,
            "description": "Efficient workhorse model with improved performance",
            "multimodal": True,
            "supports_function_calling": True,
            "supports_code_execution": True,
            "supports_search": False
        },
        
        # Gemini 2.0 Series (February 2025)
        "gemini-2.0-flash": {
            "context_length": 1000000,
            "max_output_tokens": 16000,
            "input_cost_per_1m": 0.3,
            "output_cost_per_1m": 1.2,
            "description": "Generally available Flash model with superior speed",
            "multimodal": True,
            "supports_function_calling": True,
            "supports_code_execution": True,
            "supports_search": False
        },
        "gemini-2.0-pro": {
            "context_length": 2000000,  # Largest context window available
            "max_output_tokens": 32000,
            "input_cost_per_1m": 7.0,
            "output_cost_per_1m": 21.0,
            "description": "Experimental model with strongest coding performance",
            "multimodal": True,
            "supports_function_calling": True,
            "supports_code_execution": True,
            "supports_search": True
        },
        "gemini-2.0-flash-lite": {
            "context_length": 1000000,
            "max_output_tokens": 8000,
            "input_cost_per_1m": 0.2,
            "output_cost_per_1m": 0.8,
            "description": "Better quality than 1.5 Flash at same speed and cost",
            "multimodal": True,
            "supports_function_calling": True,
            "supports_code_execution": False,
            "supports_search": False
        },
        
        # Gemini 1.5 Series (Established)
        "gemini-1.5-pro": {
            "context_length": 2000000,
            "max_output_tokens": 8192,
            "input_cost_per_1m": 3.5,
            "output_cost_per_1m": 10.5,
            "description": "High-capability model with long context",
            "multimodal": True,
            "supports_function_calling": True,
            "supports_code_execution": False,
            "supports_search": False
        },
        "gemini-1.5-flash": {
            "context_length": 1000000,
            "max_output_tokens": 8192,
            "input_cost_per_1m": 0.35,
            "output_cost_per_1m": 1.05,
            "description": "Fast and efficient model for high-volume tasks",
            "multimodal": True,
            "supports_function_calling": True,
            "supports_code_execution": False,
            "supports_search": False
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("gemini", config)
        self.api_key = config.get("api_key")
        self.default_model = config.get("default_model", "gemini-2.5-pro")
        self.region = config.get("region", "us-central1")
        
        # Statistics tracking
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.last_request_time = None
        
        if not HAS_GEMINI:
            self.logger.warning("Google Generative AI library not installed")
            self._available = False
            self.client = None
        elif not self.api_key:
            self.logger.warning("Gemini API key not provided")
            self._available = False
            self.client = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai
                self._available = True
                self.logger.info("Gemini provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {e}")
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
                "error": "Provider not configured or Gemini library not installed"
            }
        
        try:
            # Test API connection by listing models
            models = list(self.client.list_models())
            return {
                "available": True,
                "api_status": "connected",
                "models_available": len(models),
                "default_model": self.default_model,
                "region": self.region,
                "request_count": self.request_count,
                "total_tokens_used": self.total_tokens_used,
                "total_cost_usd": round(self.total_cost, 4),
                "last_request": self.last_request_time,
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models with their specifications."""
        if not self.is_available():
            return {"error": "Provider not available"}
        
        try:
            # Get models from API
            api_models = list(self.client.list_models())
            api_model_names = [model.name.replace('models/', '') for model in api_models]
            
            # Filter our known models by what's actually available
            available_models = {}
            for model_id, specs in self.MODELS.items():
                if model_id in api_model_names:
                    available_models[model_id] = specs
                else:
                    # Check for variations in naming
                    variations = [
                        model_id.replace("-", "_"),
                        model_id.replace(".", "-"),
                        f"models/{model_id}"
                    ]
                    for variation in variations:
                        if variation in api_model_names:
                            available_models[model_id] = specs
                            break
            
            return {
                "provider": "gemini",
                "models": available_models,
                "default_model": self.default_model,
                "total_count": len(available_models),
                "api_models_found": len(api_model_names)
            }
        except Exception as e:
            self.logger.error(f"Error listing Gemini models: {e}")
            return {"error": str(e)}
    
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using Gemini models."""
        if not self.is_available():
            raise ValueError("Gemini provider not available")
        
        model = model or self.default_model
        max_tokens = max_tokens or 1000
        temperature = temperature or 0.7
        
        try:
            # Initialize the model with optional system instruction
            model_kwargs = {}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            
            generation_model = self.client.GenerativeModel(model, **model_kwargs)
            
            # Configure generation settings
            generation_config = GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Safety settings (moderate blocking)
            safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                }
            ]
            
            # Generate content
            response = generation_model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Update statistics
            self.request_count += 1
            self.last_request_time = datetime.now().isoformat()
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                self.total_tokens_used += (input_tokens + output_tokens)
                
                # Calculate cost
                if model in self.MODELS:
                    model_info = self.MODELS[model]
                    input_cost = (input_tokens / 1_000_000) * model_info["input_cost_per_1m"]
                    output_cost = (output_tokens / 1_000_000) * model_info["output_cost_per_1m"]
                    self.total_cost += (input_cost + output_cost)
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error generating text with Gemini: {e}")
            raise
    
    async def get_available_tools(self) -> List[Tool]:
        """Get provider-specific tools."""
        if not self.is_available():
            return []
        
        return [
            Tool(
                name="multimodal_chat",
                description="Chat with Gemini using text and images",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt"
                        },
                        "image_data": {
                            "type": "string",
                            "description": "Base64 encoded image data (optional)"
                        },
                        "image_mime_type": {
                            "type": "string",
                            "description": "MIME type of the image (e.g., image/jpeg, image/png)",
                            "default": "image/jpeg"
                        },
                        "model": {
                            "type": "string",
                            "description": "Gemini model to use",
                            "enum": list(self.MODELS.keys())
                        },
                        "system_instruction": {
                            "type": "string",
                            "description": "System instruction for the model"
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="code_execution",
                description="Execute code using Gemini's code execution capability",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "context": {
                            "type": "string",
                            "description": "Context or explanation for the code"
                        },
                        "model": {
                            "type": "string",
                            "description": "Gemini model with code execution support",
                            "enum": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-pro"]
                        }
                    },
                    "required": ["code"]
                }
            ),
            Tool(
                name="function_calling",
                description="Use Gemini's function calling capabilities",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Prompt that may require function calls"
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
                            "description": "Available functions for the model to call"
                        },
                        "model": {
                            "type": "string",
                            "description": "Gemini model to use",
                            "enum": list(self.MODELS.keys())
                        }
                    },
                    "required": ["prompt", "functions"]
                }
            ),
            Tool(
                name="deep_think_analysis",
                description="Use Gemini 2.5 Pro's Deep Think mode for complex reasoning",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "Complex problem requiring deep reasoning"
                        },
                        "domain": {
                            "type": "string",
                            "description": "Domain of the problem (e.g., math, science, logic)",
                            "enum": ["mathematics", "science", "logic", "programming", "analysis", "general"]
                        },
                        "thinking_steps": {
                            "type": "boolean",
                            "description": "Whether to show step-by-step thinking process",
                            "default": True
                        }
                    },
                    "required": ["problem"]
                }
            )
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a provider-specific tool."""
        if not self.is_available():
            raise ValueError("Provider not available")
        
        try:
            if tool_name == "multimodal_chat":
                return await self._handle_multimodal_chat(arguments)
            elif tool_name == "code_execution":
                return await self._handle_code_execution(arguments)
            elif tool_name == "function_calling":
                return await self._handle_function_calling(arguments)
            elif tool_name == "deep_think_analysis":
                return await self._handle_deep_think_analysis(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {tool_name}")]
        except Exception as e:
            self.logger.error(f"Error calling Gemini tool {tool_name}: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _handle_multimodal_chat(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle multimodal chat with text and images."""
        prompt = arguments.get("prompt")
        image_data = arguments.get("image_data")
        image_mime_type = arguments.get("image_mime_type", "image/jpeg")
        model = arguments.get("model", self.default_model)
        system_instruction = arguments.get("system_instruction")
        
        try:
            # Initialize model
            model_kwargs = {}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            
            generation_model = self.client.GenerativeModel(model, **model_kwargs)
            
            # Prepare content
            content_parts = [prompt]
            
            if image_data:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data)
                image_part = {
                    "mime_type": image_mime_type,
                    "data": image_bytes
                }
                content_parts.append(image_part)
            
            response = generation_model.generate_content(content_parts)
            return [TextContent(type="text", text=response.text)]
            
        except Exception as e:
            raise ValueError(f"Error in multimodal chat: {str(e)}")
    
    async def _handle_code_execution(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle code execution using Gemini's capabilities."""
        code = arguments.get("code")
        context = arguments.get("context", "")
        model = arguments.get("model", "gemini-2.5-pro")
        
        # Verify model supports code execution
        if model in self.MODELS and not self.MODELS[model].get("supports_code_execution"):
            return [TextContent(type="text", text=f"Model {model} does not support code execution")]
        
        try:
            system_instruction = """You are a code execution assistant. When given code, 
            execute it step by step and provide the output. Show any intermediate results, 
            variables, and explain what the code does."""
            
            execution_prompt = f"""Execute this Python code and provide the output:

```python
{code}
```

Context: {context}

Please:
1. Execute the code step by step
2. Show any output or results
3. Explain what the code does
4. Note any potential issues or improvements
"""
            
            response = await self.generate_text(
                prompt=execution_prompt,
                model=model,
                system_instruction=system_instruction,
                max_tokens=2000
            )
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            raise ValueError(f"Error in code execution: {str(e)}")
    
    async def _handle_function_calling(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle function calling with Gemini."""
        prompt = arguments.get("prompt")
        functions = arguments.get("functions", [])
        model = arguments.get("model", self.default_model)
        
        try:
            # Convert function definitions to Gemini format
            tools = []
            for func in functions:
                tool_def = {
                    "function_declarations": [{
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": func["parameters"]
                    }]
                }
                tools.append(tool_def)
            
            generation_model = self.client.GenerativeModel(model)
            
            response = generation_model.generate_content(
                prompt,
                tools=tools
            )
            
            # Handle function calls in response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    result_text = ""
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            result_text += part.text
                        elif hasattr(part, 'function_call'):
                            # Function call detected
                            func_call = part.function_call
                            result_text += f"\n[Function Call: {func_call.name} with args: {dict(func_call.args)}]"
                    
                    return [TextContent(type="text", text=result_text or response.text)]
            
            return [TextContent(type="text", text=response.text)]
            
        except Exception as e:
            raise ValueError(f"Error in function calling: {str(e)}")
    
    async def _handle_deep_think_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle deep thinking analysis with Gemini 2.5 Pro."""
        problem = arguments.get("problem")
        domain = arguments.get("domain", "general")
        show_thinking = arguments.get("thinking_steps", True)
        
        # Only use models that support advanced reasoning
        model = "gemini-2.5-pro"
        
        try:
            system_instruction = f"""You are an expert analyst in {domain} with deep reasoning capabilities.
            When given a complex problem, think through it step by step using advanced reasoning.
            
            {"Show your thinking process clearly at each step." if show_thinking else "Provide a comprehensive analysis with clear conclusions."}
            
            Use these reasoning strategies:
            1. Break down the problem into components
            2. Identify key relationships and patterns
            3. Consider multiple perspectives and approaches
            4. Apply relevant principles and knowledge
            5. Check your reasoning for consistency
            6. Provide clear, well-supported conclusions"""
            
            analysis_prompt = f"""Problem to analyze: {problem}

Domain: {domain}

Please provide a thorough analysis using deep reasoning. {"Show your step-by-step thinking process." if show_thinking else "Focus on clear insights and conclusions."}"""
            
            response = await self.generate_text(
                prompt=analysis_prompt,
                model=model,
                system_instruction=system_instruction,
                max_tokens=4000,
                temperature=0.3  # Lower temperature for more focused reasoning
            )
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            raise ValueError(f"Error in deep think analysis: {str(e)}")
    
    async def get_available_resources(self) -> List[Resource]:
        """Get provider-specific resources."""
        if not self.is_available():
            return []
        
        return [
            Resource(
                uri="gemini://models",
                description="Information about available Gemini models",
                mimeType="application/json"
            ),
            Resource(
                uri="gemini://usage",
                description="Current usage statistics for Gemini provider",
                mimeType="application/json"
            ),
            Resource(
                uri="gemini://capabilities",
                description="Capabilities and features of Gemini models",
                mimeType="application/json"
            ),
            Resource(
                uri="gemini://pricing",
                description="Pricing information for Gemini models",
                mimeType="application/json"
            ),
            Resource(
                uri="gemini://safety",
                description="Safety settings and content filtering information",
                mimeType="application/json"
            ),
            Resource(
                uri="gemini://regions",
                description="Available regions and deployment information",
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
                    "provider": "gemini",
                    "features": {
                        "text_generation": True,
                        "multimodal": True,
                        "function_calling": True,
                        "code_execution": True,
                        "search_integration": True,
                        "long_context": True,
                        "streaming": True,
                        "safety_filtering": True
                    },
                    "model_capabilities": {
                        model: {
                            "multimodal": info.get("multimodal", False),
                            "function_calling": info.get("supports_function_calling", False),
                            "code_execution": info.get("supports_code_execution", False),
                            "search": info.get("supports_search", False),
                            "context_length": info.get("context_length"),
                            "max_output": info.get("max_output_tokens")
                        }
                        for model, info in self.MODELS.items()
                    },
                    "supported_formats": ["text", "image", "video", "audio"],
                    "programming_languages": ["python", "javascript", "java", "cpp", "go", "rust", "sql"]
                }
                return json.dumps(capabilities, indent=2)
            
            elif uri == "pricing":
                pricing_info = {
                    "provider": "gemini",
                    "currency": "USD",
                    "pricing_model": "per_million_tokens",
                    "models": {
                        model: {
                            "input_cost_per_1m_tokens": info["input_cost_per_1m"],
                            "output_cost_per_1m_tokens": info["output_cost_per_1m"],
                            "description": info["description"],
                            "multimodal_support": info.get("multimodal", False)
                        }
                        for model, info in self.MODELS.items()
                    },
                    "free_tier": {
                        "requests_per_minute": 15,
                        "requests_per_day": 1500,
                        "tokens_per_minute": 32000
                    }
                }
                return json.dumps(pricing_info, indent=2)
            
            elif uri == "safety":
                safety_info = {
                    "provider": "gemini",
                    "safety_categories": [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH", 
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT"
                    ],
                    "threshold_levels": [
                        "BLOCK_NONE",
                        "BLOCK_ONLY_HIGH",
                        "BLOCK_MEDIUM_AND_ABOVE",
                        "BLOCK_LOW_AND_ABOVE"
                    ],
                    "default_settings": {
                        "harassment": "BLOCK_MEDIUM_AND_ABOVE",
                        "hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
                        "sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE",
                        "dangerous_content": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    "content_filtering": {
                        "enabled": True,
                        "configurable": True,
                        "bypass_available": False
                    }
                }
                return json.dumps(safety_info, indent=2)
            
            elif uri == "regions":
                regions_info = {
                    "provider": "gemini",
                    "current_region": self.region,
                    "available_regions": [
                        "us-central1",
                        "us-east1", 
                        "us-west1",
                        "europe-west1",
                        "asia-southeast1"
                    ],
                    "region_features": {
                        "us-central1": {
                            "all_models": True,
                            "code_execution": True,
                            "search_integration": True,
                            "latency": "low"
                        },
                        "europe-west1": {
                            "all_models": True,
                            "code_execution": True,
                            "search_integration": False,
                            "latency": "medium"
                        }
                    }
                }
                return json.dumps(regions_info, indent=2)
            
            else:
                return json.dumps({"error": f"Unknown resource: {uri}"})
                
        except Exception as e:
            self.logger.error(f"Error getting Gemini resource {uri}: {e}")
            return json.dumps({"error": str(e)})
    
    async def estimate_cost(self, prompt: str, max_tokens: int = 1000, model: Optional[str] = None) -> Dict[str, Any]:
        """Estimate the cost of a request."""
        model = model or self.default_model
        
        if model not in self.MODELS:
            return {"error": f"Unknown model: {model}"}
            
        # Simple token estimation (roughly 4 characters per token)
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = max_tokens
        
        model_info = self.MODELS[model]
        input_cost = (estimated_input_tokens / 1_000_000) * model_info["input_cost_per_1m"]
        output_cost = (estimated_output_tokens / 1_000_000) * model_info["output_cost_per_1m"]
        total_cost = input_cost + output_cost
        
        return {
            "provider": "gemini",
            "model": model,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "currency": "USD",
            "pricing_model": "per_million_tokens",
            "multimodal_supported": model_info.get("multimodal", False)
        }
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this provider."""
        return {
            "provider": "gemini",
            "default_model": self.default_model,
            "region": self.region,
            "status": "active" if self.is_available() else "inactive",
            "statistics": {
                "total_requests": self.request_count,
                "total_tokens_used": self.total_tokens_used,
                "total_cost_usd": round(self.total_cost, 4),
                "last_request": self.last_request_time,
                "average_cost_per_request": round(self.total_cost / max(self.request_count, 1), 4)
            },
            "configuration": {
                "api_configured": bool(self.api_key),
                "models_available": len(self.MODELS),
                "multimodal_enabled": True,
                "safety_filtering": True
            },
            "capabilities_summary": {
                "text_generation": True,
                "image_understanding": True,
                "code_execution": True,
                "function_calling": True,
                "long_context": True
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.last_request_time = None
        self.logger.info("Gemini provider statistics reset")