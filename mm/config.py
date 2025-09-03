"""
Configuration module that now imports from sheet_config.py.
This maintains backward compatibility while using the new Google Sheets-based configuration.
"""

from .sheet_config import MMConfig, load_config

__all__ = ['MMConfig', 'load_config']

