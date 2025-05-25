"""
Configuration Management for Universal MCP Server
Handles loading and validation of configuration from environment variables and config files
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Configuration manager for the Universal MCP Server."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables from .env file
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            self.logger.info("Loaded environment variables from .env file")
        
        # Load configuration file if provided
        self.file_config = {}
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.file_config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config file {config_file}: {e}")
        
        # Validate required configurations
        self._validate_config()
    
    def _validate_config(self):
        """Validate that at least one provider is configured."""
        providers_configured = []
        
        if self.get_openai_config().get("api_key"):
            providers_configured.append("OpenAI")
        if self.get_gemini_config().get("api_key"):
            providers_configured.append("Gemini")
        if self.get_anthropic_config().get("api_key"):
            providers_configured.append("Anthropic")
        
        if not providers_configured:
            self.logger.warning("No LLM providers are configured. Please set API keys in environment variables or config file.")
        else:
            self.logger.info(f"Configured providers: {', '.join(providers_configured)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with fallback order: env var -> config file -> default."""
        # Try environment variable first
        env_value = os.getenv(key.upper())
        if env_value is not None:
            return env_value
        
        # Try config file
        if key in self.file_config:
            return self.file_config[key]
        
        # Return default
        return default
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            "api_key": self.get("openai_api_key"),
            "base_url": self.get("openai_base_url", "https://api.openai.com/v1"),
            "organization": self.get("openai_organization"),
            "default_model": self.get("openai_default_model", "gpt-4o"),
            "max_retries": int(self.get("openai_max_retries", "3")),
            "timeout": int(self.get("openai_timeout", "60"))
        }
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Google Gemini configuration."""
        return {
            "api_key": self.get("gemini_api_key"),
            "base_url": self.get("gemini_base_url", "https://generativelanguage.googleapis.com/v1beta"),
            "default_model": self.get("gemini_default_model", "gemini-1.5-pro"),
            "max_retries": int(self.get("gemini_max_retries", "3")),
            "timeout": int(self.get("gemini_timeout", "60")),
            "region": self.get("gemini_region", "us-central1")
        }
    
    def get_anthropic_config(self) -> Dict[str, Any]:
        """Get Anthropic Claude configuration."""
        return {
            "api_key": self.get("anthropic_api_key"),
            "base_url": self.get("anthropic_base_url", "https://api.anthropic.com"),
            "default_model": self.get("anthropic_default_model", "claude-3-5-sonnet-20241022"),
            "max_retries": int(self.get("anthropic_max_retries", "3")),
            "timeout": int(self.get("anthropic_timeout", "60"))
        }
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration."""
        return {
            "host": self.get("server_host", "localhost"),
            "port": int(self.get("server_port", "8080")),
            "debug": self.get("server_debug", "false").lower() == "true",
            "log_level": self.get("log_level", "INFO").upper(),
            "max_concurrent_requests": int(self.get("max_concurrent_requests", "100")),
            "request_timeout": int(self.get("request_timeout", "300")),
            "enable_cors": self.get("enable_cors", "true").lower() == "true"
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "enable_encryption": self.get("enable_encryption", "true").lower() == "true",
            "enable_audit_logging": self.get("enable_audit_logging", "true").lower() == "true",
            "secret_key": self.get("secret_key", self._generate_secret_key()),
            "rate_limiting": {
                "enabled": self.get("rate_limiting_enabled", "true").lower() == "true",
                "requests_per_minute": int(self.get("rate_limit_requests_per_minute", "60")),
                "requests_per_hour": int(self.get("rate_limit_requests_per_hour", "1000"))
            },
            "authentication": {
                "enabled": self.get("auth_enabled", "false").lower() == "true",
                "type": self.get("auth_type", "api_key"),  # api_key, oauth, jwt
                "api_keys": self.get("auth_api_keys", "").split(",") if self.get("auth_api_keys") else []
            }
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return {
            "enabled": self.get("ui_enabled", "true").lower() == "true",
            "title": self.get("ui_title", "Universal MCP Server"),
            "theme": self.get("ui_theme", "modern"),
            "show_provider_status": self.get("ui_show_provider_status", "true").lower() == "true",
            "show_metrics": self.get("ui_show_metrics", "true").lower() == "true",
            "refresh_interval": int(self.get("ui_refresh_interval", "30"))
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.get("log_level", "INFO").upper(),
            "file": self.get("log_file", "logs/server.log"),
            "max_size_mb": int(self.get("log_max_size_mb", "100")),
            "backup_count": int(self.get("log_backup_count", "5")),
            "format": self.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            "enable_console": self.get("log_enable_console", "true").lower() == "true"
        }
    
    def _generate_secret_key(self) -> str:
        """Generate a random secret key if none is provided."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return all configuration as a dictionary (without sensitive data)."""
        config = {
            "server": self.get_server_config(),
            "ui": self.get_ui_config(),
            "logging": self.get_logging_config(),
            "providers": {
                "openai": {
                    "configured": bool(self.get_openai_config().get("api_key")),
                    "default_model": self.get_openai_config().get("default_model")
                },
                "gemini": {
                    "configured": bool(self.get_gemini_config().get("api_key")),
                    "default_model": self.get_gemini_config().get("default_model")
                },
                "anthropic": {
                    "configured": bool(self.get_anthropic_config().get("api_key")),
                    "default_model": self.get_anthropic_config().get("default_model")
                }
            }
        }
        
        # Add security config without sensitive data
        security_config = self.get_security_config()
        config["security"] = {
            "encryption_enabled": security_config["enable_encryption"],
            "audit_logging_enabled": security_config["enable_audit_logging"],
            "rate_limiting": security_config["rate_limiting"],
            "authentication": {
                "enabled": security_config["authentication"]["enabled"],
                "type": security_config["authentication"]["type"]
            }
        }
        
        return config
    
    def save_sample_config(self, filename: str = "config.json.sample"):
        """Save a sample configuration file."""
        sample_config = {
            "openai_api_key": "your-openai-api-key-here",
            "gemini_api_key": "your-gemini-api-key-here", 
            "anthropic_api_key": "your-anthropic-api-key-here",
            "openai_default_model": "gpt-4o",
            "gemini_default_model": "gemini-1.5-pro",
            "anthropic_default_model": "claude-3-5-sonnet-20241022",
            "server_host": "localhost",
            "server_port": 8080,
            "log_level": "INFO",
            "ui_enabled": True,
            "ui_title": "Universal MCP Server",
            "enable_encryption": True,
            "enable_audit_logging": True,
            "rate_limiting_enabled": True,
            "rate_limit_requests_per_minute": 60,
            "auth_enabled": False
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(sample_config, f, indent=2)
            self.logger.info(f"Sample configuration saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving sample config: {e}")
    
    def validate_provider_config(self, provider: str) -> bool:
        """Validate configuration for a specific provider."""
        if provider == "openai":
            config = self.get_openai_config()
            return bool(config.get("api_key"))
        elif provider == "gemini":
            config = self.get_gemini_config()
            return bool(config.get("api_key"))
        elif provider == "anthropic":
            config = self.get_anthropic_config()
            return bool(config.get("api_key"))
        else:
            return False
    
    def get_log_file(self) -> str:
        """Get the log file path."""
        return self.get_logging_config()["file"]
    
    def get_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all provider configurations."""
        return {
            "openai": self.get_openai_config(),
            "gemini": self.get_gemini_config(),
            "anthropic": self.get_anthropic_config()
        }
    
    def update_config(self, key: str, value: Any) -> None:
        """Update a configuration value (for runtime changes)."""
        self.file_config[key] = value
    
    def get_model_defaults(self) -> Dict[str, str]:
        """Get default models for all providers."""
        return {
            "openai": self.get_openai_config()["default_model"],
            "gemini": self.get_gemini_config()["default_model"],
            "anthropic": self.get_anthropic_config()["default_model"]
        }
    
    def is_provider_configured(self, provider: str) -> bool:
        """Check if a provider is configured with an API key."""
        return self.validate_provider_config(provider)
    
    def get_configured_providers(self) -> List[str]:
        """Get list of configured provider names."""
        configured = []
        for provider in ["openai", "gemini", "anthropic"]:
            if self.is_provider_configured(provider):
                configured.append(provider)
        return configured
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration for backup or transfer."""
        config = self.to_dict()
        
        if include_secrets:
            # WARNING: This includes sensitive information
            config["secrets"] = {
                "openai_api_key": self.get("openai_api_key"),
                "gemini_api_key": self.get("gemini_api_key"),
                "anthropic_api_key": self.get("anthropic_api_key"),
                "secret_key": self.get("secret_key")
            }
        
        return config