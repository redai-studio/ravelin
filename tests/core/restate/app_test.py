import pytest
from unittest.mock import AsyncMock, MagicMock

from ravelin.core.restate.app import NON_RETRYABLE_ERRORS, create_restate_app
from ravelin.streaming.event import LIFECYCLE_ERROR


class TestCreateRestateApp:
    """Test create_restate_app factory function."""

    def test_returns_workflow_and_service(self):
        """create_restate_app should return a Restate Workflow and Service."""
        pytest.importorskip("restate", reason="restate_sdk required")
        from restate import Workflow, Service

        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        pipeline = RestatePipeline()
        io_handler = PipelineIOHandler(inputs={}, outputs={})

        workflow, executor = create_restate_app(pipeline, io_handler, "TestPipeline")

        assert isinstance(workflow, Workflow)
        assert isinstance(executor, Service)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "envelope, message",
        [
            ([], "Expected an object"),
            ({"payload": {}}, "awakeable_id"),
            ({"awakeable_id": 42, "payload": {}}, "awakeable_id"),
            ({"awakeable_id": "  ", "payload": {}}, "awakeable_id"),
            ({"awakeable_id": "", "payload": {}}, "awakeable_id"),
            ({"awakeable_id": "awk-1"}, "payload is required"),
        ],
    )
    async def test_resolve_rejects_invalid_envelope_without_resolving(self, envelope, message):
        import restate
        from ravelin.core.restate import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        _, executor = create_restate_app(RestatePipeline(), PipelineIOHandler(inputs={}, outputs={}), "ResolveTest")
        ctx = MagicMock()
        with pytest.raises(restate.TerminalError, match=message):
            await executor.handlers["resolve_awakeable"].fn(ctx, envelope)
        ctx.resolve_awakeable.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [{}, None, False, 0, "", [], {"response": "yes", "extra": 1}])
    async def test_resolve_forwards_explicit_json_payload(self, payload):
        from ravelin.core.restate import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        _, executor = create_restate_app(RestatePipeline(), PipelineIOHandler(inputs={}, outputs={}), "ResolveTest")
        ctx = MagicMock()
        result = await executor.handlers["resolve_awakeable"].fn(ctx, {"awakeable_id": "awk-1", "payload": payload})
        ctx.resolve_awakeable.assert_called_once_with("awk-1", payload)
        assert result == {"resolved": True, "awakeable_id": "awk-1"}

    def test_passes_invocation_retry_policy_to_workflow_and_executor(self):
        """create_restate_app should pass through explicit Restate retry policy."""
        pytest.importorskip("restate", reason="restate_sdk required")
        import restate

        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        pipeline = RestatePipeline()
        io_handler = PipelineIOHandler(inputs={}, outputs={})
        policy = restate.InvocationRetryPolicy(max_attempts=3, on_max_attempts="pause")

        workflow, executor = create_restate_app(
            pipeline,
            io_handler,
            "PolicyPipeline",
            invocation_retry_policy=policy,
        )

        assert workflow.invocation_retry_policy is policy
        assert executor.invocation_retry_policy is policy

    @pytest.mark.asyncio
    async def test_non_recoverable_component_error_becomes_terminal_with_cause(self):
        pytest.importorskip("restate", reason="restate_sdk required")
        import restate
        from haystack import component

        from ravelin.core.restate import NonRecoverableError
        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        @component
        class BadComponent:
            @component.output_types(result=str)
            def run(self, text: str) -> dict[str, str]:
                raise NonRecoverableError("corrupt state")

        pipeline = RestatePipeline()
        pipeline.add_component("bad", BadComponent())
        create_restate_app(pipeline, PipelineIOHandler(inputs={}, outputs={}), "BadPipeline")

        ctx = MagicMock()
        ctx.request.return_value.headers = {}

        with pytest.raises(restate.TerminalError) as exc_info:
            await pipeline._executor_handler(  # pylint: disable=protected-access
                ctx,
                {"component_name": "bad", "inputs": {"text": "hello"}, "workflow_id": "wf-1", "call_index": 0},
            )

        assert "non_recoverable" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, NonRecoverableError)

    @pytest.mark.asyncio
    async def test_generic_component_value_error_remains_retryable(self):
        pytest.importorskip("restate", reason="restate_sdk required")
        from haystack import component

        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        @component
        class BadComponent:
            @component.output_types(result=str)
            def run(self, text: str) -> dict[str, str]:
                raise ValueError("ordinary bug")

        pipeline = RestatePipeline()
        pipeline.add_component("bad", BadComponent())
        create_restate_app(pipeline, PipelineIOHandler(inputs={}, outputs={}), "RetryablePipeline")

        ctx = MagicMock()
        ctx.request.return_value.headers = {}

        with pytest.raises(ValueError, match="ordinary bug"):
            await pipeline._executor_handler(  # pylint: disable=protected-access
                ctx,
                {"component_name": "bad", "inputs": {"text": "hello"}, "workflow_id": "wf-1", "call_index": 0},
            )

    @pytest.mark.asyncio
    async def test_component_error_journal_event_is_structured(self):
        pytest.importorskip("restate", reason="restate_sdk required")
        from haystack import component

        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        @component
        class BadComponent:
            @component.output_types(result=str)
            def run(self, text: str) -> dict[str, str]:
                raise ValueError("ordinary bug")

        pipeline = RestatePipeline()
        pipeline.add_component("bad", BadComponent())
        backend = MagicMock()
        backend.write_event = AsyncMock()
        backend.close_run = AsyncMock()
        journal_events = []

        async def journal_hook(event):
            journal_events.append(event)

        create_restate_app(
            pipeline,
            PipelineIOHandler(inputs={}, outputs={}),
            "StructuredErrorPipeline",
            streaming_backend=backend,
            on_journal_event=journal_hook,
        )

        ctx = MagicMock()
        ctx.request.return_value.headers = {}

        with pytest.raises(ValueError):
            await pipeline._executor_handler(  # pylint: disable=protected-access
                ctx,
                {"component_name": "bad", "inputs": {"text": "hello"}, "workflow_id": "wf-1", "call_index": 3},
            )

        error_events = [event for event in journal_events if event.subtype == LIFECYCLE_ERROR]
        assert len(error_events) == 1
        payload = error_events[0].payload
        assert payload["category"] == "unknown_retryable"
        assert payload["label"] == "unclassified_exception"
        assert payload["terminal"] is False
        assert payload["retryable"] is True
        assert payload["component"] == "bad"
        assert payload["workflow_id"] == "wf-1"
        assert payload["call_index"] == 3
        assert "ValueError" in payload["stack"]
        assert str(payload["fingerprint"]).startswith("bad:unclassified_exception:")


class TestWorkflowCloseRun:
    """The workflow closes the run (writes the stream sentinel) iff the invocation
    is genuinely over — success or TerminalError. Suspension and retryable errors
    resume the same invocation and must leave the stream open."""

    @staticmethod
    def _make_run_handler(pipeline_run_side_effect):
        pytest.importorskip("restate", reason="restate_sdk required")

        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        pipeline = RestatePipeline()
        pipeline.run = AsyncMock(side_effect=pipeline_run_side_effect)
        backend = MagicMock()
        backend.close_run = AsyncMock()
        workflow, _ = create_restate_app(
            pipeline,
            PipelineIOHandler(inputs={}, outputs={}),
            "CloseRunPipeline",
            streaming_backend=backend,
        )
        ctx = MagicMock()
        ctx.key.return_value = "run-xyz"
        ctx.request.return_value.headers = {}
        return workflow.handlers["run"].fn, ctx, backend

    @pytest.mark.asyncio
    async def test_success_closes_run(self):
        run_fn, ctx, backend = self._make_run_handler(lambda *a, **k: {})

        await run_fn(ctx, {})

        backend.close_run.assert_awaited_once_with("run-xyz")

    @pytest.mark.asyncio
    async def test_suspension_does_not_close_run(self):
        """Suspension (awakeable gate) must not write the run-over sentinel — the
        run is paused, not finished. Reconnecting clients would otherwise hit the
        sentinel and never see events emitted after the gate resolves."""
        from restate.exceptions import SuspendedException

        run_fn, ctx, backend = self._make_run_handler(SuspendedException())

        with pytest.raises(SuspendedException):
            await run_fn(ctx, {})

        backend.close_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_terminal_error_closes_run(self):
        import restate

        run_fn, ctx, backend = self._make_run_handler(restate.TerminalError("boom"))

        with pytest.raises(restate.TerminalError):
            await run_fn(ctx, {})

        backend.close_run.assert_awaited_once_with("run-xyz")

    @pytest.mark.asyncio
    async def test_retryable_error_does_not_close_run(self):
        """A retryable (non-terminal) error escaping to the handler will be retried
        by Restate from where it left off — the invocation is not over, so no
        sentinel. Writing one would strand it before the post-retry events."""
        run_fn, ctx, backend = self._make_run_handler(ConnectionError("transient"))

        with pytest.raises(ConnectionError):
            await run_fn(ctx, {})

        backend.close_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_internal_retry_signal_does_not_close_run(self):
        """Restate's own retryable-error signal (SdkInternalException) resumes the
        invocation and must not close the run."""
        from restate.exceptions import SdkInternalException

        run_fn, ctx, backend = self._make_run_handler(SdkInternalException())

        with pytest.raises(SdkInternalException):
            await run_fn(ctx, {})

        backend.close_run.assert_not_called()


class TestNonRetryableErrors:
    """Test the NON_RETRYABLE_ERRORS configuration."""

    def test_non_retryable_errors_contains_expected_types(self):
        """NON_RETRYABLE_ERRORS should contain known programming errors."""
        expected = {
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            IndexError,
            AssertionError,
            ImportError,
            SyntaxError,
        }
        assert expected.issubset(set(NON_RETRYABLE_ERRORS))

    def test_non_retryable_errors_is_tuple(self):
        """NON_RETRYABLE_ERRORS should be a tuple (for except clause)."""
        assert isinstance(NON_RETRYABLE_ERRORS, tuple)

    def test_all_entries_are_exception_types(self):
        """All entries should be subclasses of Exception."""
        for error_type in NON_RETRYABLE_ERRORS:
            assert issubclass(error_type, BaseException)

    def test_can_be_used_in_except_clause(self):
        """Should be usable in an except clause."""

        def raise_type_error():
            raise TypeError("test error")

        with pytest.raises(TypeError):
            try:
                raise_type_error()
            except NON_RETRYABLE_ERRORS:
                raise

    def test_catches_all_configured_errors(self):
        """Each configured error type should be caught."""
        for error_type in NON_RETRYABLE_ERRORS:
            caught = False
            try:
                raise error_type("test")
            except NON_RETRYABLE_ERRORS:
                caught = True
            assert caught, f"{error_type.__name__} was not caught"

    def test_does_not_catch_other_errors(self):
        """Should not catch errors not in the list."""

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            try:
                raise CustomError("test")
            except NON_RETRYABLE_ERRORS:
                pass  # Should not be caught
