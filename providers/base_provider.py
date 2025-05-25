"""
Base Provider Class for Universal MCP Server
Abstract base class that all LLM providers must implement
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mcp.types import Resource, Tool, TextContent

class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._available = False
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the provider."""
        pass
    
    @abstractmethod
    async def list_models(self) -> Dict[str, Any]:
        """List available models for this provider."""
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using the provider's models."""
        pass
    
    @abstractmethod
    async def get_available_tools(self) -> List[Tool]:
        """Get provider-specific tools."""
        pass
    
    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a provider-specific tool."""
        pass
    
    @abstractmethod
    async def get_available_resources(self) -> List[Resource]:
        """Get provider-specific resources."""
        pass
    
    @abstractmethod
    async def get_resource(self, uri: str) -> str:
        """Get a provider-specific resource."""
        pass
    
    @abstractmethod
    async def estimate_cost(self, prompt: str, max_tokens: int = 1000, model: Optional[str] = None) -> Dict[str, Any]:
        """Estimate the cost of a request."""
        pass
    
    @abstractmethod
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this provider."""
        pass
    
    def get_name(self) -> str:
        """Get the provider name."""
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """Get the provider configuration."""
        return self.config.copy()
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(f"[{self.name}] {message}")
    
    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(f"[{self.name}] {message}")
