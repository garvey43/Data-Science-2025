"""
Password Vault package initialization.

This package provides:
- Vault: a class to handle encrypted password storage
- CLI commands: add, get, delete, list
"""

from .vault import Vault

__all__ = ["Vault"]
