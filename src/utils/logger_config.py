"""
Utility module for logging and configuration management.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml


def setup_logging(
    log_dir: str = "./logs",
    level: str = "INFO",
    name: str = "stock_explainable"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save logs
        level: Logging level
        name: Logger name
        
    Returns:
        Configured logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log")
    )
    file_handler.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    config = _replace_env_vars(config)
    return config


def _replace_env_vars(config: Any) -> Any:
    """Recursively replace ${VAR_NAME} with environment variables."""
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        if config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
    return config


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories from config."""
    for path_key, path_val in config.get('paths', {}).items():
        Path(path_val).mkdir(parents=True, exist_ok=True)
