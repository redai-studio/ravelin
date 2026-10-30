"""
This module provides a collection of classes that represent and manage tool services in the application.

A tool service is a deployed tool (or a selection of tools) that can be used for various downstream tasks.
Tool services can be used to perform tasks such as web searches, data processing, and often provide an auxiliary
role in the application.
"""

from ravelin.assets.tool_services._base import (
    _ToolService as ToolService,
    TOOL_SERVICE_REGISTRY,
    register_tool_service,
)
from ravelin.assets.tool_services.webscraper_services import *  # noqa: F403
from ravelin.assets.tool_services.websearch_services import *  # noqa: F403

__all__ = [
    "ToolService",
    "TOOL_SERVICE_REGISTRY",
    "register_tool_service",
]
