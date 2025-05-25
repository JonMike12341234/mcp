"""
Provider Package for Universal MCP Server
Initializes and exports all LLM provider implementations
"""

from .base_provider import BaseProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .anthropic_provider import AnthropicProvider

# Export all providers
__all__ = [
    'BaseProvider',
    'OpenAIProvider', 
    'GeminiProvider',
    'AnthropicProvider'
]

# Provider registry for dynamic loading
PROVIDER_REGISTRY = {
    'openai': OpenAIProvider,
    'gemini': GeminiProvider,
    'anthropic': AnthropicProvider
}

def get_provider_class(provider_name: str):
    """Get a provider class by name."""
    return PROVIDER_REGISTRY.get(provider_name.lower())

def list_available_providers():
    """List all available provider names."""
    return list(PROVIDER_REGISTRY.keys())

def create_provider(provider_name: str, config: dict):
    """Create a provider instance by name."""
    provider_class = get_provider_class(provider_name)
    if provider_class:
        return provider_class(config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
