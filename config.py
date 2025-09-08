#!/usr/bin/env python3
"""
Configuration Management System
Implements debug vs production modes for better debugging and development
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Configuration manager that separates debug and production settings.
    Follows the principle of "moving bugs left" by making debug mode fail fast
    and provide maximum information.
    """

    def __init__(self, config_file: str = "grader_config.json"):
        self.config_file = Path(config_file)
        self.debug_mode = self._detect_debug_mode()
        self.config = self._load_config()
        self._setup_logging()

    def _detect_debug_mode(self) -> bool:
        """Detect if we're in debug mode based on environment and context"""
        # Check environment variable
        if os.getenv('DEBUG_MODE', '').lower() in ('true', '1', 'yes'):
            return True

        # Check for development indicators
        if os.getenv('PYTHONPATH') or os.getcwd().endswith('dev'):
            return True

        # Check if we're running from source (not installed)
        if Path('grade_assignments.py').exists():
            return True

        return False

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with debug/production overrides"""
        base_config = self._get_base_config()

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                base_config.update(file_config)
            except (json.JSONDecodeError, IOError) as e:
                self._log_error(f"Failed to load config file: {e}")
                # Continue with base config

        # Apply debug/production overrides
        if self.debug_mode:
            base_config.update(self._get_debug_overrides())
        else:
            base_config.update(self._get_production_overrides())

        return base_config

    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration that works for both modes"""
        return {
            "grading_weights": {
                "syntax_correctness": 30,
                "code_quality": 25,
                "documentation": 15,
                "functionality": 20,
                "style": 10
            },
            "ai_detection": {
                "enabled": True,
                "strict_mode": False,
                "penalty_high": 30,
                "penalty_very_high": 50
            },
            "file_extensions": [".py", ".ipynb"],
            "excluded_files": ["instruction.md", "desktop.ini", "__pycache__"],
            "feedback_settings": {
                "generate_individual_files": True,
                "include_code_suggestions": True,
                "include_ai_warnings": True
            },
            "automation": {
                "auto_grade_delay": 1800,
                "max_concurrent_jobs": 3,
                "notification_enabled": True
            },
            "debug": {
                "verbose_logging": False,
                "fail_fast": False,
                "extra_validation": False,
                "memory_debugging": False
            }
        }

    def _get_debug_overrides(self) -> Dict[str, Any]:
        """Debug mode overrides - fail fast, maximum information"""
        return {
            "ai_detection": {
                "enabled": True,
                "strict_mode": True,  # More aggressive AI detection
                "penalty_high": 40,   # Higher penalties for debugging
                "penalty_very_high": 60
            },
            "debug": {
                "verbose_logging": True,
                "fail_fast": True,      # Stop on first error
                "extra_validation": True,  # Additional checks
                "memory_debugging": True
            },
            "automation": {
                "auto_grade_delay": 300,  # Faster feedback in debug
                "max_concurrent_jobs": 1,  # Sequential for debugging
                "notification_enabled": True
            }
        }

    def _get_production_overrides(self) -> Dict[str, Any]:
        """Production mode overrides - graceful handling, performance"""
        return {
            "ai_detection": {
                "enabled": True,
                "strict_mode": False,  # Less aggressive for production
                "penalty_high": 20,
                "penalty_very_high": 40
            },
            "debug": {
                "verbose_logging": False,
                "fail_fast": False,
                "extra_validation": False,
                "memory_debugging": False
            },
            "automation": {
                "auto_grade_delay": 1800,  # Standard delay
                "max_concurrent_jobs": 5,   # Parallel processing
                "notification_enabled": False  # Less noise in production
            }
        }

    def _setup_logging(self):
        """Setup logging based on debug mode"""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        if self.get('debug.verbose_logging', False):
            log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('debug.log' if self.debug_mode else 'grader.log')
            ]
        )

    def _log_error(self, message: str):
        """Log error with appropriate level"""
        logger = logging.getLogger(__name__)
        logger.error(message)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """Set configuration value with dot notation support"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def is_debug_mode(self) -> bool:
        """Check if we're in debug mode"""
        return self.debug_mode

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            self._log_error(f"Failed to save config: {e}")

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about current configuration"""
        return {
            "debug_mode": self.debug_mode,
            "config_file": str(self.config_file),
            "config_keys": list(self.config.keys()),
            "environment": {
                "DEBUG_MODE": os.getenv('DEBUG_MODE'),
                "PYTHONPATH": os.getenv('PYTHONPATH'),
                "cwd": os.getcwd()
            }
        }

# Global configuration instance
config = ConfigManager()

if __name__ == "__main__":
    # Debug info when run directly
    print("=== Configuration Debug Info ===")
    debug_info = config.get_debug_info()
    for key, value in debug_info.items():
        print(f"{key}: {value}")

    print("\n=== Current Configuration ===")
    print(json.dumps(config.config, indent=2))