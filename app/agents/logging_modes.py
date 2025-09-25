# app/agents/logging_modes.py
"""
Easy logging mode configuration for different use cases.
Set environment variables or call functions to control logging detail.
"""
import os
from typing import Dict, List, Optional
from enum import Enum

class LoggingMode(Enum):
    """Predefined logging modes for different scenarios."""
    SILENT = "silent"           # Only errors
    MINIMAL = "minimal"         # Just LLM requests/responses  
    STANDARD = "standard"       # LLM + tools + agent actions
    VERBOSE = "verbose"         # Everything except prompts
    DEBUG = "debug"            # Everything including full prompts

class LoggingConfig:
    """Centralized logging configuration."""
    
    MODES: Dict[LoggingMode, Dict[str, str]] = {
        LoggingMode.SILENT: {
            "LOG_LEVEL": "ERROR",
            "LOG_VERBOSE": "0",
            "LOG_PROMPT": "0", 
            "LOG_TOOLS": "0",
            "LOG_CHAINS": "0",
            "LOG_HTTP": "0"
        },
        LoggingMode.MINIMAL: {
            "LOG_LEVEL": "INFO",
            "LOG_VERBOSE": "0", 
            "LOG_PROMPT": "0",
            "LOG_TOOLS": "0",
            "LOG_CHAINS": "0", 
            "LOG_HTTP": "1"  # Only HTTP requests (LLM calls)
        },
        LoggingMode.STANDARD: {
            "LOG_LEVEL": "INFO",
            "LOG_VERBOSE": "0",
            "LOG_PROMPT": "0",
            "LOG_TOOLS": "1",
            "LOG_CHAINS": "1",
            "LOG_HTTP": "1"
        },
        LoggingMode.VERBOSE: {
            "LOG_LEVEL": "DEBUG", 
            "LOG_VERBOSE": "1",  # Agent executor stdout
            "LOG_PROMPT": "0",
            "LOG_TOOLS": "1",
            "LOG_CHAINS": "1", 
            "LOG_HTTP": "1"
        },
        LoggingMode.DEBUG: {
            "LOG_LEVEL": "DEBUG",
            "LOG_VERBOSE": "1",
            "LOG_PROMPT": "1",  # Full prompts
            "LOG_TOOLS": "1", 
            "LOG_CHAINS": "1",
            "LOG_HTTP": "1"
        }
    }

    @classmethod
    def set_mode(cls, mode: LoggingMode) -> None:
        """Set environment variables for the specified logging mode."""
        config = cls.MODES[mode]
        for key, value in config.items():
            os.environ[key] = value
        print(f"Logging mode set to: {mode.value}")
        cls.print_current_config()

    @classmethod
    def get_current_mode(cls) -> Optional[LoggingMode]:
        """Detect current logging mode based on environment variables."""
        current_config = {
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"), 
            "LOG_VERBOSE": os.getenv("LOG_VERBOSE", "0"),
            "LOG_PROMPT": os.getenv("LOG_PROMPT", "0"),
            "LOG_TOOLS": os.getenv("LOG_TOOLS", "1"),
            "LOG_CHAINS": os.getenv("LOG_CHAINS", "1"),
            "LOG_HTTP": os.getenv("LOG_HTTP", "1")
        }
        
        for mode, config in cls.MODES.items():
            if all(current_config.get(k, "0") == v for k, v in config.items()):
                return mode
        return None  # Custom configuration

    @classmethod
    def print_current_config(cls) -> None:
        """Print current logging configuration."""
        mode = cls.get_current_mode()
        if mode:
            print(f"Current mode: {mode.value}")
        else:
            print("Current mode: custom")
        
        config_vars = ["LOG_LEVEL", "LOG_VERBOSE", "LOG_PROMPT", "LOG_TOOLS", "LOG_CHAINS", "LOG_HTTP"]
        print("Configuration:")
        for var in config_vars:
            value = os.getenv(var, "not set")
            print(f"  {var}={value}")

def set_logging_mode(mode_name: str) -> None:
    """Set logging mode by string name."""
    try:
        mode = LoggingMode(mode_name.lower())
        LoggingConfig.set_mode(mode)
    except ValueError:
        available = [m.value for m in LoggingMode]
        print(f"Invalid mode '{mode_name}'. Available modes: {available}")

# Convenience functions
def silent_mode():
    """Only log errors."""
    LoggingConfig.set_mode(LoggingMode.SILENT)

def minimal_mode(): 
    """Only log LLM HTTP requests/responses."""
    LoggingConfig.set_mode(LoggingMode.MINIMAL)

def standard_mode():
    """Log LLM calls, tools, and agent actions (default)."""
    LoggingConfig.set_mode(LoggingMode.STANDARD)

def verbose_mode():
    """Log everything except full prompts."""
    LoggingConfig.set_mode(LoggingMode.VERBOSE)

def debug_mode():
    """Log everything including full prompts.""" 
    LoggingConfig.set_mode(LoggingMode.DEBUG)

# Environment detection helper
def should_log_component(component: str) -> bool:
    """Check if a specific component should be logged."""
    component_map = {
        "verbose": "LOG_VERBOSE",
        "prompt": "LOG_PROMPT", 
        "tools": "LOG_TOOLS",
        "chains": "LOG_CHAINS",
        "http": "LOG_HTTP"
    }
    
    env_var = component_map.get(component.lower())
    if not env_var:
        return True  # Default to logging unknown components
    
    return os.getenv(env_var, "0").lower() in ("1", "true", "yes")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python logging_modes.py [silent|minimal|standard|verbose|debug|status]")
        print("\nAvailable modes:")
        for mode in LoggingMode:
            print(f"  {mode.value}")
        print("  status - show current configuration")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "status":
        LoggingConfig.print_current_config()
    elif command in [m.value for m in LoggingMode]:
        set_logging_mode(command)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)