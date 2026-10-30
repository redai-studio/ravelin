"""
This module initializes the `ravelin` package and sets up the environment.
"""

import os

from ravelin.version import __version__
from ravelin.core.pipeline import AsyncPipeline, Pipeline

# Disable Haystack telemetry by setting the environment variable
# See https://docs.haystack.deepset.ai/docs/telemetry for more information
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

__all__ = ["__version__", "Pipeline", "AsyncPipeline"]
