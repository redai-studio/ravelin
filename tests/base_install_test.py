"""Verify public base imports in a fresh interpreter with optional SDKs unavailable."""

import os
import subprocess
import sys
import textwrap


def test_base_imports_without_redis_or_restate():
    """A full development environment must not conceal eager optional imports."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys

        original_find_spec = importlib.util.find_spec

        def find_spec(name, package=None):
            if name.split(".")[0] in {"redis", "restate"}:
                return None
            return original_find_spec(name, package)

        class BlockExtras(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"redis", "restate"}:
                    raise ModuleNotFoundError("Optional SDK unavailable", name=fullname)
                return None

        importlib.util.find_spec = find_spec
        sys.meta_path.insert(0, BlockExtras())

        from ravelin import Pipeline
        from ravelin.components.agents import Agent
        from ravelin.components.generators import ChatGenerator
        from ravelin.components.tools import RestateToolInvoker
        from ravelin.core.restate.context import has_restate_context
        from ravelin.streaming.backends import RedisStreamingBackend

        assert not has_restate_context()
        assert "redis" not in sys.modules
        assert "restate" not in sys.modules
        try:
            RedisStreamingBackend()
        except ImportError as exc:
            assert "ravelin[redis]" in str(exc)
        else:
            raise AssertionError("Redis construction must require its optional dependency")
        try:
            RestateToolInvoker(tools=[])
        except ImportError as exc:
            assert "restate_sdk[serde]" in str(exc)
        else:
            raise AssertionError("Restate construction must require its optional dependency")
        print("base imports OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "HAYSTACK_TELEMETRY_ENABLED": "False"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "base imports OK" in result.stdout
