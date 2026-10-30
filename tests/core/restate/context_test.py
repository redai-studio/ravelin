import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from ravelin.core.restate.context import _restate_ctx, get_restate_context, has_restate_context


class TestContextVarIsolation:
    """Test that ContextVar provides proper isolation."""

    def test_get_restate_context_raises_without_context(self):
        """Should raise RuntimeError when called outside workflow execution."""
        # Ensure context is not set
        token = _restate_ctx.set(None)
        try:
            with pytest.raises(RuntimeError, match="No Restate context available"):
                get_restate_context()
        finally:
            _restate_ctx.reset(token)

    def test_get_restate_context_returns_set_context(self):
        """Should return the context that was set."""
        mock_ctx = MagicMock()
        token = _restate_ctx.set(mock_ctx)
        try:
            result = get_restate_context()
            assert result is mock_ctx
        finally:
            _restate_ctx.reset(token)

    def test_has_restate_context(self):
        token = _restate_ctx.set(None)
        try:
            assert has_restate_context() is False
        finally:
            _restate_ctx.reset(token)

        mock_ctx = MagicMock()
        token = _restate_ctx.set(mock_ctx)
        try:
            assert has_restate_context() is True
        finally:
            _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_contextvar_isolation_between_tasks(self):
        """ContextVar should be isolated between different asyncio tasks."""
        results: dict[str, Any] = {}
        mock_ctx_a = MagicMock()
        mock_ctx_b = MagicMock()

        async def task_a():
            token = _restate_ctx.set(mock_ctx_a)
            try:
                await asyncio.sleep(0.01)  # Yield to other task
                results["task_a"] = get_restate_context()
            finally:
                _restate_ctx.reset(token)

        async def task_b():
            token = _restate_ctx.set(mock_ctx_b)
            try:
                await asyncio.sleep(0.01)  # Yield to other task
                results["task_b"] = get_restate_context()
            finally:
                _restate_ctx.reset(token)

        await asyncio.gather(task_a(), task_b())

        # Each task should see its own context (not the other's)
        assert results["task_a"] is mock_ctx_a
        assert results["task_b"] is mock_ctx_b
        assert results["task_a"] is not results["task_b"]

    @pytest.mark.asyncio
    async def test_contextvar_propagates_to_nested_coroutines(self):
        """ContextVar should propagate to nested async calls."""
        mock_ctx = MagicMock(name="outer_ctx")

        async def inner_func():
            # Should see the context set by outer
            return get_restate_context()

        async def middle_func():
            return await inner_func()

        async def outer_func():
            token = _restate_ctx.set(mock_ctx)
            try:
                return await middle_func()
            finally:
                _restate_ctx.reset(token)

        result = await outer_func()
        assert result is mock_ctx


class SimpleTestComponent:
    """A simple component for testing that captures the Restate context."""

    def __init__(self):
        self.captured_context = None
        self.run_count = 0

    def run(self, text: str) -> dict[str, Any]:
        self.run_count += 1
        try:
            self.captured_context = get_restate_context()
        except RuntimeError:
            self.captured_context = None
        return {"output": f"processed: {text}"}


class AsyncTestComponent:
    """An async component for testing."""

    __haystack_supports_async__ = True

    def __init__(self):
        self.captured_context = None
        self.run_count = 0

    async def run_async(self, text: str) -> dict[str, Any]:
        self.run_count += 1
        try:
            self.captured_context = get_restate_context()
        except RuntimeError:
            self.captured_context = None
        return {"output": f"async processed: {text}"}


class TestContextPropagationInComponents:
    """Test that Restate context propagates into component execution."""

    @pytest.mark.asyncio
    async def test_context_not_available_in_sync_component(self):
        """Context should NOT be available in sync component (can't await in sync code)."""
        # Sync components run in executor without context propagation
        # This is intentional - sync code can't use Restate ctx anyway

        component = SimpleTestComponent()
        mock_ctx = MagicMock(name="workflow_ctx")

        # Set the context before running
        token = _restate_ctx.set(mock_ctx)
        try:
            # Simulate running in executor WITHOUT context propagation
            # (matches the actual implementation)
            loop = asyncio.get_running_loop()

            result = await loop.run_in_executor(None, lambda: component.run("test"))

            assert result == {"output": "processed: test"}
            # The component should NOT have captured the context
            assert component.captured_context is None
        finally:
            _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_context_available_in_async_component(self):
        """Context should be available in async component."""
        component = AsyncTestComponent()
        mock_ctx = MagicMock(name="workflow_ctx")

        token = _restate_ctx.set(mock_ctx)
        try:
            result = await component.run_async("test")

            assert result == {"output": "async processed: test"}
            assert component.captured_context is mock_ctx
        finally:
            _restate_ctx.reset(token)
