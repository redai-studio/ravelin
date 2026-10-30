"""
This module imports Haystack's Tool and ComponentTool classes, as well as the create_tool_from_function and tool
decorator in the Ravelin namespace. It offers a quick way to access these classes and functions without needing to
import them directly from their respective modules.

However, this module might be removed once the extra functionalities of from_function are fully integrated into the
Haystack library.
"""

# ruff: noqa: I001 (ignore import order as we need to import Tool before ComponentTool)
from haystack.tools.component_tool import ComponentTool
from haystack.tools.serde_utils import serialize_tools_or_toolset, deserialize_tools_or_toolset_inplace
from haystack.tools.tool import Tool, _check_duplicate_tool_names

from ravelin.tools.from_function import create_tool_from_function, tool

__all__ = [
    "Tool",
    "_check_duplicate_tool_names",
    "serialize_tools_or_toolset",
    "deserialize_tools_or_toolset_inplace",
    "create_tool_from_function",
    "tool",
    "ComponentTool",
]
