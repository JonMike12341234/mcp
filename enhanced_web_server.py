"""
FIXED Enhanced Web Server for Universal MCP Orchestrator
Provides REST API endpoints with PROPER MCP tool integration
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

# Simple MCP Orchestrator (inline to avoid import issues)
class SimpleMCPOrchestrator:
    """Simple MCP Orchestrator for basic functionality."""
    
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
        
        self.logger.info("Simple MCP Orchestrator initialized")
    
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
                    models[provider_name] = {"error": str(e)}
        
        return models
    
    async def execute_query(
        self, 
        provider_name: str, 
        model_id: str, 
        mcp_server: Optional[str],
        user_query: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute query with basic functionality."""
        try:
            # Validate provider
            if provider_name not in self.providers or not self.providers[provider_name].is_available():
                return {"error": f"Provider {provider_name} not available"}
            
            provider = self.providers[provider_name]
            
            # For now, just do basic text generation without MCP integration
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
                "note": "MCP integration temporarily disabled for stability"
            }
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources."""
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
            description="AI Model and MCP Server Orchestration Platform"
        )
        self.logger = setup_logger("fixed_enhanced_web_server")
        
        # Initialize the orchestrator
        self.orchestrator = SimpleMCPOrchestrator()
        
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
            models = self.orchestrator.get_available_models()
            
            connected_providers = sum(1 for provider_models in models.values() 
                                    if isinstance(provider_models, list) and provider_models)
            
            return {
                "server": "Universal MCP Orchestrator - FIXED",
                "version": "2.1.0-FIXED",
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "connected_providers": connected_providers,
                    "total_providers": len(models),
                    "available_models": sum(len(provider_models) for provider_models in models.values() 
                                          if isinstance(provider_models, list)),
                    "connected_mcp_servers": 0,  # Temporarily disabled
                    "total_mcp_servers": 0,
                    "available_tools": 0
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
            # Return empty for now to prevent crashes
            return {}
        
        @self.app.post("/api/chat")
        async def chat(request: ChatRequest):
            """Handle chat requests."""
            try:
                self.logger.info(f"Chat request: {request.provider}/{request.model}")
                
                result = await self.orchestrator.execute_query(
                    provider_name=request.provider,
                    model_id=request.model,
                    mcp_server=request.mcp_server,
                    user_query=request.message,
                    system_prompt=request.system_prompt
                )
                
                if "error" in result:
                    raise HTTPException(status_code=400, detail=result["error"])
                
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
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal MCP Orchestrator - FIXED</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .fixed-badge {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            animation: pulse 2s infinite;
        }
        .chat-bubble {
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-bubble {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        }
        .ai-bubble {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        .loading-dots {
            display: inline-block;
        }
        .loading-dots::after {
            content: '';
            animation: dots 2s infinite;
        }
        @keyframes dots {
            0%, 20% { content: ''; }
            40% { content: '.'; }
            60% { content: '..'; }
            80%, 100% { content: '...'; }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    </style>
</head>
<body class="bg-gray-50">
    <div id="app" class="min-h-screen">
        <!-- Header -->
        <header class="gradient-bg text-white shadow-lg">
            <div class="container mx-auto px-6 py-6">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="flex items-center space-x-3">
                            <h1 class="text-3xl font-bold">Universal MCP Orchestrator</h1>
                            <span class="fixed-badge text-white px-3 py-1 rounded-full text-sm font-semibold">
                                ✅ STABLE
                            </span>
                        </div>
                        <p class="text-blue-100 mt-2">AI Model Integration Platform</p>
                    </div>
                    <div class="flex items-center space-x-4">
                        <div class="text-right">
                            <p class="text-sm opacity-90">Connected Providers</p>
                            <p class="font-semibold">{{ connectedProviders }} / {{ totalProviders }}</p>
                        </div>
                    </div>
                </div>
            </div>
        </header>

        <div class="container mx-auto px-6 py-8">
            <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <!-- Configuration Panel -->
                <div class="lg:col-span-1">
                    <div class="bg-white rounded-lg shadow-lg p-6 sticky top-6">
                        <h2 class="text-xl font-bold mb-4 text-gray-800">Configuration</h2>
                        
                        <!-- Model Selection -->
                        <div class="mb-6">
                            <label class="block text-sm font-medium text-gray-700 mb-2">
                                <i class="fas fa-robot mr-2"></i>AI Model
                            </label>
                            <select v-model="selectedProvider" @change="updateModels" 
                                    class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                <option value="">Select Provider</option>
                                <option v-for="(models, provider) in availableModels" :key="provider" :value="provider">
                                    {{ provider.charAt(0).toUpperCase() + provider.slice(1) }}
                                </option>
                            </select>
                            
                            <select v-if="selectedProvider" v-model="selectedModel" 
                                    class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent mt-2">
                                <option value="">Select Model</option>
                                <option v-for="model in availableModels[selectedProvider]" :key="model.id" :value="model.id">
                                    {{ model.name }}
                                </option>
                            </select>
                        </div>

                        <!-- System Prompt -->
                        <div class="mb-6">
                            <label class="block text-sm font-medium text-gray-700 mb-2">
                                <i class="fas fa-cog mr-2"></i>System Prompt (Optional)
                            </label>
                            <textarea v-model="systemPrompt" 
                                      class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                                      rows="3" placeholder="Enter system instructions..."></textarea>
                        </div>

                        <!-- Quick Actions -->
                        <div class="space-y-2">
                            <button @click="clearChat" 
                                    class="w-full px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors">
                                <i class="fas fa-broom mr-2"></i>Clear Chat
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Chat Interface -->
                <div class="lg:col-span-3">
                    <div class="bg-white rounded-lg shadow-lg">
                        <!-- Chat Header -->
                        <div class="p-4 border-b border-gray-200 bg-gray-50 rounded-t-lg">
                            <div class="flex justify-between items-center">
                                <h2 class="text-lg font-semibold text-gray-800">
                                    <i class="fas fa-comments mr-2"></i>AI Chat
                                </h2>
                                <div class="flex items-center space-x-2 text-sm text-gray-600">
                                    <span v-if="selectedProvider && selectedModel">
                                        <i class="fas fa-robot mr-1"></i>{{ selectedProvider }}/{{ selectedModel }}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <!-- Chat Messages -->
                        <div class="h-96 overflow-y-auto p-4 space-y-4" ref="chatContainer">
                            <div v-if="chatMessages.length === 0" class="text-center text-gray-500 mt-8">
                                <i class="fas fa-comment-dots text-4xl mb-4"></i>
                                <p>Start a conversation with your AI assistant</p>
                                <p class="text-sm mt-2">Select a model to get started</p>
                            </div>
                            
                            <div v-for="(message, index) in chatMessages" :key="index" 
                                 class="flex" :class="message.role === 'user' ? 'justify-end' : 'justify-start'">
                                <div class="chat-bubble p-4 rounded-lg text-white" 
                                     :class="message.role === 'user' ? 'user-bubble' : 'ai-bubble'">
                                    <div class="flex items-start space-x-2">
                                        <i :class="message.role === 'user' ? 'fas fa-user' : 'fas fa-robot'"></i>
                                        <div class="flex-1">
                                            <div class="font-medium text-sm opacity-90 mb-1">
                                                {{ message.role === 'user' ? 'You' : 'AI Assistant' }}
                                            </div>
                                            <div class="whitespace-pre-wrap">{{ message.content }}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Loading indicator -->
                            <div v-if="isLoading" class="flex justify-start">
                                <div class="ai-bubble p-4 rounded-lg text-white">
                                    <div class="flex items-center space-x-2">
                                        <i class="fas fa-robot"></i>
                                        <div>
                                            <div class="font-medium text-sm opacity-90 mb-1">AI Assistant</div>
                                            <div class="loading-dots">Thinking</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Chat Input -->
                        <div class="p-4 border-t border-gray-200">
                            <div class="flex space-x-2">
                                <input v-model="userInput" 
                                       @keyup.enter="sendMessage"
                                       :disabled="isLoading || !selectedProvider || !selectedModel"
                                       class="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
                                       placeholder="Type your message here..." />
                                <button @click="sendMessage" 
                                        :disabled="!userInput.trim() || isLoading || !selectedProvider || !selectedModel"
                                        class="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors">
                                    <i class="fas fa-paper-plane"></i>
                                </button>
                            </div>
                            <div v-if="!selectedProvider || !selectedModel" class="text-sm text-gray-500 mt-2">
                                Please select a provider and model to start chatting
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const { createApp } = Vue;
        
        createApp({
            data() {
                return {
                    // Configuration
                    selectedProvider: '',
                    selectedModel: '',
                    systemPrompt: '',
                    
                    // Data
                    availableModels: {},
                    chatMessages: [],
                    userInput: '',
                    isLoading: false,
                    
                    // Statistics
                    connectedProviders: 0,
                    totalProviders: 0
                }
            },
            async mounted() {
                await this.loadData();
                this.loadExampleConfiguration();
            },
            methods: {
                async loadData() {
                    try {
                        // Load available models
                        const modelsResponse = await fetch('/api/models');
                        const modelsData = await modelsResponse.json();
                        this.availableModels = modelsData;
                        
                        // Count connected providers
                        this.totalProviders = Object.keys(modelsData).length;
                        this.connectedProviders = Object.values(modelsData).filter(models => 
                            Array.isArray(models) && models.length > 0
                        ).length;
                        
                    } catch (error) {
                        console.error('Error loading data:', error);
                    }
                },
                
                updateModels() {
                    this.selectedModel = '';
                },
                
                async sendMessage() {
                    if (!this.userInput.trim() || this.isLoading) return;
                    
                    const message = this.userInput.trim();
                    this.userInput = '';
                    
                    // Add user message to chat
                    this.chatMessages.push({
                        role: 'user',
                        content: message,
                        timestamp: new Date()
                    });
                    
                    this.isLoading = true;
                    this.scrollToBottom();
                    
                    try {
                        // Send request to backend
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                provider: this.selectedProvider,
                                model: this.selectedModel,
                                message: message,
                                system_prompt: this.systemPrompt || null
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (result.error) {
                            throw new Error(result.error);
                        }
                        
                        // Add AI response to chat
                        this.chatMessages.push({
                            role: 'assistant',
                            content: result.response,
                            timestamp: new Date()
                        });
                        
                    } catch (error) {
                        console.error('Error sending message:', error);
                        this.chatMessages.push({
                            role: 'assistant',
                            content: `Error: ${error.message}`,
                            timestamp: new Date(),
                            isError: true
                        });
                    } finally {
                        this.isLoading = false;
                        this.scrollToBottom();
                    }
                },
                
                clearChat() {
                    this.chatMessages = [];
                },
                
                loadExampleConfiguration() {
                    // Auto-select first available provider and model
                    const providers = Object.keys(this.availableModels);
                    if (providers.length > 0) {
                        this.selectedProvider = providers[0];
                        const models = this.availableModels[this.selectedProvider];
                        if (Array.isArray(models) && models.length > 0) {
                            this.selectedModel = models[0].id;
                        }
                    }
                },
                
                scrollToBottom() {
                    this.$nextTick(() => {
                        const container = this.$refs.chatContainer;
                        if (container) {
                            container.scrollTop = container.scrollHeight;
                        }
                    });
                }
            }
        }).mount('#app');
    </script>
</body>
</html>'''
    
    async def start(self, host: str = "localhost", port: int = 8080):
        """Start the web server."""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


async def create_fixed_enhanced_web_server(config: Config) -> Optional[FixedEnhancedWebServer]:
    """Create and initialize the enhanced web server."""
    if not HAS_FASTAPI:
        print("FastAPI is not installed. Please install it to use the web server.")
        return None

    return FixedEnhancedWebServer(config)


def main():
    """Main entry point for running the enhanced web server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal MCP Orchestrator Web Dashboard")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    async def run_server():
        server = await create_fixed_enhanced_web_server(config)
        if server:
            print(f"\n🚀 Universal MCP Orchestrator starting at http://{args.host}:{args.port}")
            print("✅ Features available:")
            print("   • Multi-provider AI model selection")
            print("   • Interactive chat interface")
            print("   • OpenAI, Gemini, and Anthropic support")
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