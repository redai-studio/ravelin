from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

import pytest
import restate
from restate.server_context import ServerDurableFuture, ServerInvocationContext

from ravelin.core.restate.approvals import require_approval, ApprovalTimeout
from ravelin.core.restate import ToolExecutionApprovalPayload
from ravelin.core.restate.context import _restate_ctx


@dataclass
class HumanResponse:
    response: str


def _context_with_resolved_bytes(raw):
    """Mock transport notifications; retain SDK serde selection and futures."""
    ctx = MagicMock()
    # Notifications contain plain serde bytes, with no journal codec configured.
    ctx.journal_codec = None
    ctx.vm.sys_awakeable.return_value = ("test-awakeable-id", 1)
    ctx.vm.is_completed.side_effect = lambda handle: handle == 1
    ctx.must_take_notification = AsyncMock(return_value=raw)
    ctx.create_future.side_effect = lambda handle, serde: ServerDurableFuture(
        ctx, handle, ServerInvocationContext._create_fetch_result_coroutine(ctx, handle, serde)
    )
    ctx.awakeable.side_effect = lambda **kwargs: ServerInvocationContext.awakeable(ctx, **kwargs)
    ctx.sleep.return_value = ServerDurableFuture(ctx, 2, AsyncMock())
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [None, 30])
@pytest.mark.parametrize("raw", [b'{"response":"yes"}', b'{"response":"yes","extra":1}'])
async def test_approval_decodes_real_bytes(raw, timeout_seconds):
    ctx = _context_with_resolved_bytes(raw)
    token = _restate_ctx.set(ctx)
    try:
        assert await require_approval(HumanResponse, timeout_seconds=timeout_seconds) == HumanResponse("yes")
    finally:
        _restate_ctx.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [None, 30])
@pytest.mark.parametrize("raw", [b"{}", b'{"response":42}', b"null", b"[]", b"invalid-json", b""])
async def test_invalid_approval_bytes_are_terminal(raw, timeout_seconds):
    ctx = _context_with_resolved_bytes(raw)
    token = _restate_ctx.set(ctx)
    try:
        with pytest.raises(restate.TerminalError, match="invalid payload"):
            await require_approval(HumanResponse, timeout_seconds=timeout_seconds)
    finally:
        _restate_ctx.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", ["false", 1, {}, None])
async def test_tool_approval_rejects_non_boolean_decisions(approved):
    import json

    ctx = _context_with_resolved_bytes(json.dumps({"approved": approved}).encode())
    token = _restate_ctx.set(ctx)
    try:
        with pytest.raises(restate.TerminalError, match="invalid payload"):
            await require_approval()
    finally:
        _restate_ctx.reset(token)


@pytest.mark.asyncio
async def test_approval_serde_retains_wire_encoding():
    from restate.serde import DefaultSerde

    ctx = _context_with_resolved_bytes(b'{"response":"yes"}')
    token = _restate_ctx.set(ctx)
    try:
        await require_approval(HumanResponse)
        serde = ctx.awakeable.call_args.kwargs["serde"]
        assert serde.serialize(HumanResponse("yes")) == DefaultSerde(HumanResponse).serialize(HumanResponse("yes"))
    finally:
        _restate_ctx.reset(token)


@pytest.mark.asyncio
async def test_custom_decoder_terminal_error_is_preserved():
    error = restate.TerminalError("custom validation")

    @dataclass
    class CustomResponse:
        response: str

        @classmethod
        def from_dict(cls, data):
            raise error

    ctx = _context_with_resolved_bytes(b'{"response":"yes"}')
    token = _restate_ctx.set(ctx)
    try:
        with pytest.raises(restate.TerminalError) as caught:
            await require_approval(CustomResponse)
        assert caught.value is error
    finally:
        _restate_ctx.reset(token)


@pytest.mark.asyncio
async def test_notifier_failure_keeps_its_original_exception():
    ctx = _context_with_resolved_bytes(b'{"response":"yes"}')
    ctx.run_typed = AsyncMock(side_effect=ValueError("notifier failed"))
    token = _restate_ctx.set(ctx)
    try:
        with pytest.raises(ValueError, match="notifier failed"):
            await require_approval(HumanResponse, notifier=AsyncMock())
    finally:
        _restate_ctx.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [restate.TerminalError("rejected"), restate.exceptions.SuspendedException()])
async def test_approval_preserves_rejection_and_suspension(exception):
    ctx = _context_with_resolved_bytes(b"{}")
    ctx.must_take_notification.side_effect = exception
    token = _restate_ctx.set(ctx)
    try:
        with pytest.raises(type(exception)) as caught:
            await require_approval(HumanResponse)
        assert caught.value is exception
    finally:
        _restate_ctx.reset(token)


class TestRequireApprovalFunction:
    """Test the require_approval utility function."""

    @pytest.mark.asyncio
    async def test_require_approval_raises_without_context(self):
        """Should raise RuntimeError when called without Restate context."""
        # Ensure no context is set
        token = _restate_ctx.set(None)
        try:
            with pytest.raises(RuntimeError, match="No Restate context available"):
                await require_approval()
        finally:
            _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_require_approval_creates_awakeable(self):
        """Should create an awakeable and return its result as ToolExecutionApprovalPayload."""
        mock_ctx = MagicMock()
        mock_promise = AsyncMock()
        mock_promise.return_value = ToolExecutionApprovalPayload(approved=True, reason="approved by test_user")

        # Mock awakeable to return id and promise
        mock_ctx.awakeable.return_value = ("test-awakeable-id", mock_promise())

        token = _restate_ctx.set(mock_ctx)
        try:
            result = await require_approval()

            mock_ctx.awakeable.assert_called_once()
            assert isinstance(result, ToolExecutionApprovalPayload)
            assert result.approved is True
        finally:
            _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_require_approval_with_notifier(self):
        """Should call notifier with awakeable notification."""
        mock_ctx = MagicMock()
        mock_promise = AsyncMock()
        mock_promise.return_value = ToolExecutionApprovalPayload(approved=True)
        mock_ctx.awakeable.return_value = ("test-awakeable-id", mock_promise())

        # Mock run_typed for the notifier call
        mock_ctx.run_typed = AsyncMock()

        notifier_called_with = None

        async def test_notifier(notification):
            nonlocal notifier_called_with
            notifier_called_with = notification

        token = _restate_ctx.set(mock_ctx)
        try:
            await require_approval(
                metadata={"request_id": "123"},
                notifier=test_notifier,
            )

            # run_typed should have been called for the notifier
            mock_ctx.run_typed.assert_called_once()
            call_args = mock_ctx.run_typed.call_args

            # First arg should be the name
            assert call_args[0][0] == "notify_approval"
            # Second arg should be the notifier
            assert call_args[0][1] is test_notifier
            # Fourth arg should be the notification
            notification = call_args[0][3]
            assert notification["awakeable_id"] == "test-awakeable-id"
            assert notification["metadata"] == {"request_id": "123"}
        finally:
            _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_require_approval_with_custom_run_options(self):
        """Should pass custom RunOptions to run_typed."""
        pytest.importorskip("restate", reason="restate_sdk required")
        from restate import RunOptions

        mock_ctx = MagicMock()
        mock_promise = AsyncMock()
        mock_promise.return_value = ToolExecutionApprovalPayload(approved=True)
        mock_ctx.awakeable.return_value = ("test-awakeable-id", mock_promise())
        mock_ctx.run_typed = AsyncMock()

        custom_options = RunOptions(max_attempts=5)

        async def test_notifier(notification):
            pass

        token = _restate_ctx.set(mock_ctx)
        try:
            await require_approval(
                notifier=test_notifier,
                notifier_run_options=custom_options,
            )

            # run_typed should have been called with custom options
            call_args = mock_ctx.run_typed.call_args
            assert call_args[0][2] is custom_options
        finally:
            _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_require_approval_with_timeout_approved(self):
        """Should return result when approved before timeout expires."""
        pytest.importorskip("restate", reason="restate_sdk required")
        import restate
        from ravelin.core.restate.approvals import ToolExecutionApprovalPayload

        mock_ctx = MagicMock()
        mock_promise = AsyncMock()
        mock_sleep = AsyncMock()

        mock_ctx.awakeable.return_value = ("test-awakeable-id", mock_promise)
        mock_ctx.sleep.return_value = mock_sleep

        # Mock restate.select to return approval result
        approval_result = ToolExecutionApprovalPayload(approved=True, reason="Looks good")
        with patch.object(restate, "select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = ["approval", approval_result]

            token = _restate_ctx.set(mock_ctx)
            try:
                result = await require_approval(timeout_seconds=30)

                assert result.approved is True
                assert result.reason == "Looks good"
            finally:
                _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_require_approval_raises_timeout(self):
        """Should raise ApprovalTimeout when timeout expires."""
        pytest.importorskip("restate", reason="restate_sdk required")
        import restate

        mock_ctx = MagicMock()
        mock_promise = AsyncMock()
        mock_sleep = AsyncMock()

        mock_ctx.awakeable.return_value = ("test-awakeable-id", mock_promise)
        mock_ctx.sleep.return_value = mock_sleep

        # Mock restate.select to return timeout
        with patch.object(restate, "select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = ["timeout", None]

            token = _restate_ctx.set(mock_ctx)
            try:
                with pytest.raises(ApprovalTimeout) as exc_info:
                    await require_approval(timeout_seconds=30)

                assert exc_info.value.timeout_seconds == 30
                assert exc_info.value.awakeable_id == "test-awakeable-id"
            finally:
                _restate_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_require_approval_invalid_payload_raises_terminal_error(self):
        """Non-ToolExecutionApprovalPayload awakeable result should raise TerminalError."""
        pytest.importorskip("restate", reason="restate_sdk required")
        import restate

        mock_ctx = MagicMock()
        mock_promise = AsyncMock()
        mock_sleep = AsyncMock()

        mock_ctx.awakeable.return_value = ("test-awakeable-id", mock_promise)
        mock_ctx.sleep.return_value = mock_sleep

        # Mock restate.select to return the wrong type
        with patch.object(restate, "select", new_callable=AsyncMock) as mock_select:
            mock_select.return_value = ["approval", {"approved": True}]  # dict, not ToolExecutionApprovalPayload

            token = _restate_ctx.set(mock_ctx)
            try:
                with pytest.raises(restate.TerminalError, match="invalid payload"):
                    await require_approval(timeout_seconds=30)
            finally:
                _restate_ctx.reset(token)
