import os
import unittest
from typing import Optional, List, Callable
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import httpx
from haystack.utils import Secret

from ravelin.assets._base import resolve_secret
from ravelin.assets.asset_factory import ModelServiceFactory
from ravelin.assets.model_services import register_model_service
from ravelin.assets.model_services.chat_model_services import HttpxChatModelService
from ravelin.assets.model_services.chat_model_services.openai import _convert_message_to_openai_format
from ravelin.core.restate.context import _restate_ctx
from ravelin.dataclasses import ChatMessage, StreamingChunk
from ravelin.dataclasses.chat_message import TextContent


class CustomizedHttpxChatModelService(HttpxChatModelService):
    def build_request(
        self,
        client,
        messages,
        *,
        model=None,
        timeout=None,
        stream=False,
        tools=None,
        inference_kwargs=None,
        extra_headers=None,
        **kwargs,
    ):
        inference_kwargs = inference_kwargs or {}
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        return client.build_request(
            "POST",
            url=self.base_url,
            json={
                "messages": [_convert_message_to_openai_format(message) for message in messages],
                "model": model,
                "stream": stream,
                "tools": tools,
                **inference_kwargs,
            },
            headers=headers,
            timeout=timeout,
            **kwargs,
        )

    @staticmethod
    def _parse_message(raw_message):
        return ChatMessage(_content=[TextContent(raw_message["content"])], _role=raw_message["role"], _name=None)

    def parse_response(self, response, *, stream=False, streaming_callbacks=None):
        raw_messages = response.json().get("messages", [])
        return [self._parse_message(raw_message) for raw_message in raw_messages]

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
        streaming_callbacks: Optional[List[Callable[[StreamingChunk], None]]] = None,
    ) -> List[ChatMessage]:
        raw_messages = response.json().get("messages", [])
        return [self._parse_message(raw_message) for raw_message in raw_messages]


class TestCustomizedChatModelService(unittest.TestCase):
    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Register our custom model service
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_initialization(self):
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url", api_key=Secret.from_token("abc")
        )
        self.assertEqual("test_customized_model_service", customized_model_service.name)
        self.assertEqual(
            {
                "base_url": "test_url",
                "organization": None,
                "inference_api_format": "custom_chat_format",
            },
            customized_model_service.configs,
        )
        self.assertEqual("abc", customized_model_service.secrets["api_key"].resolve_value())

    def test_initialization_with_secret_api_key(self):
        secret = Secret.from_env_var("CUSTOM_CHAT_API_KEY")
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url", api_key=secret
        )
        # The Secret is stored as-is; the getter returns the raw value.
        self.assertIs(secret, customized_model_service.api_key)
        self.assertEqual({"api_key": secret}, customized_model_service.secrets)
        with patch.dict(os.environ, {"CUSTOM_CHAT_API_KEY": "resolved-key"}):
            self.assertEqual("resolved-key", resolve_secret(customized_model_service.api_key))

    def test_secret_api_key_survives_dict_round_trip(self):
        secret = Secret.from_env_var("CUSTOM_CHAT_API_KEY")
        service = CustomizedHttpxChatModelService(name="svc", base_url="test_url", api_key=secret)
        restored = ModelServiceFactory.load_from_dict(service.to_dict())
        self.assertIsInstance(restored.api_key, Secret)
        with patch.dict(os.environ, {"CUSTOM_CHAT_API_KEY": "resolved-key"}):
            self.assertEqual("resolved-key", resolve_secret(restored.api_key))

    def test_infer(self):
        messages = [ChatMessage.from_user("Hello world!")]
        # Patch the httpx client to return a response with embeddings
        with patch("httpx.Client") as mock_client:
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )
            mock_request = httpx.Request("POST", "https://example.com")
            mock_client.return_value.build_request.return_value = mock_request

            # Create a proper mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "messages": [{"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "World!"}]
            }
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.send.return_value = mock_response

            replies = customized_model_service.infer(
                messages, model="custom-model", timeout=5.0, stream=True, tools=[], inference_kwargs={"param": "value"}
            )
            # Check that the client was called with the expected arguments
            mock_client.assert_called_once_with()
            # Check that the request was sent
            mock_client.return_value.build_request.assert_called_once_with(
                "POST",
                url="test_url",
                json={
                    "messages": [{"role": "user", "content": "Hello world!"}],
                    "model": "custom-model",
                    "stream": True,
                    "tools": [],
                    "param": "value",
                },
                headers={},
                timeout=5.0,
            )
            mock_client.return_value.send.assert_called_once_with(mock_request, stream=True)
            mock_response.raise_for_status.assert_called_once_with()
            # Check that the response was parsed correctly
            self.assertEqual(2, len(replies))
            self.assertEqual(ChatMessage.from_user("Hello?"), replies[0])
            self.assertEqual(ChatMessage.from_assistant("World!"), replies[1])

    def test_infer_error(self):
        messages = [ChatMessage.from_user("Hello world!")]
        mock_response = httpx.Response(status_code=500, content="test error")
        # Patch the httpx client to raise an error
        with patch("httpx.Client") as mock_client:
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )
            mock_request = httpx.Request("POST", "https://example.com")
            mock_client.return_value.build_request.return_value = mock_request
            mock_client.return_value.send.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error: '500 Internal Server Error' for url: 'test_url'", request=Mock(), response=mock_response
            )
            with self.assertRaises(httpx.HTTPStatusError) as context:
                customized_model_service.infer(
                    messages, model="custom-model", timeout=5.0, stream=True, inference_kwargs={"param": "value"}
                )
            self.assertEqual(mock_response, context.exception.response)
            self.assertEqual(
                (
                    "Server error: '500 Internal Server Error' for url: 'test_url'\n"
                    "Model Service: test_customized_model_service\n"
                    "Details: test error"
                ),
                str(context.exception),
            )

    def test_creates_reusable_client_lazily(self):
        with patch("httpx.Client") as mock_client_cls:
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )
            mock_client_cls.assert_not_called()

            self.assertIs(mock_client_cls.return_value, customized_model_service._client)
            self.assertIs(mock_client_cls.return_value, customized_model_service._client)
            mock_client_cls.assert_called_once_with()

    def test_close_closes_reusable_client(self):
        with patch("httpx.Client") as mock_client_cls:
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )
            mock_client = customized_model_service._client

            customized_model_service.close()

            mock_client.close.assert_called_once_with()
            # The next use creates a new client
            customized_model_service._client  # pylint: disable=pointless-statement
            self.assertEqual(2, mock_client_cls.call_count)


class TestCustomizedChatModelServiceAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Register our custom model service
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    async def test_ainfer(self):
        messages = [ChatMessage.from_user("Hello world!")]
        # Patch the httpx client to return a response with embeddings
        with patch("httpx.AsyncClient") as mock_client_context:
            # Patch the response from the client
            mock_client = MagicMock()
            mock_client.is_closed = False
            mock_client_context.return_value = mock_client
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )

            # Set up request
            mock_request = httpx.Request("POST", "https://example.com")
            mock_client.build_request.return_value = mock_request

            # Create the mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "messages": [{"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "World!"}]
            }

            # IMPORTANT: Make send an async method
            async def async_send(*args, **kwargs):
                # Capture the arguments for later assertion
                mock_client.send.call_args_list.append((args, kwargs))
                return mock_response

            mock_client.send = async_send
            # Initialize the call_args_list for tracking
            mock_client.send.call_args_list = []

            replies = await customized_model_service.ainfer(
                messages, model="custom-model", timeout=5.0, stream=True, tools=[], inference_kwargs={"param": "value"}
            )
            # Check that the client was called with the expected arguments
            mock_client_context.assert_called_once_with()
            # Check that the request was sent
            mock_client.build_request.assert_called_once_with(
                "POST",
                url="test_url",
                json={
                    "messages": [{"role": "user", "content": "Hello world!"}],
                    "model": "custom-model",
                    "stream": True,
                    "tools": [],
                    "param": "value",
                },
                headers={},
                timeout=5.0,
            )
            # Check that send was called with the right arguments
            self.assertEqual(1, len(mock_client.send.call_args_list))
            args, kwargs = mock_client.send.call_args_list[0]
            self.assertEqual(mock_request, args[0])
            self.assertEqual(True, kwargs.get("stream"))
            # Check that raise_for_status is called
            mock_response.raise_for_status.assert_called_once_with()
            # Check that the response was parsed correctly
            self.assertEqual(2, len(replies))
            self.assertEqual(ChatMessage.from_user("Hello?"), replies[0])
            self.assertEqual(ChatMessage.from_assistant("World!"), replies[1])

    async def test_ainfer_error(self):
        messages = [ChatMessage.from_user("Hello world!")]

        # Create a mock request object
        mock_request = MagicMock(name="request")

        # Create a real Response object for the error
        mock_response = httpx.Response(status_code=500, content=b"test error", request=mock_request)

        with patch("httpx.AsyncClient") as mock_client_context:
            # Use a regular MagicMock as the base
            mock_client = MagicMock()
            mock_client.is_closed = False
            mock_client_context.return_value = mock_client
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )

            # Set up build_request as a SYNCHRONOUS method
            mock_client.build_request.return_value = mock_request

            # Create the HTTPStatusError to be raised
            http_error = httpx.HTTPStatusError(
                "Server error: '500 Internal Server Error' for url: 'test_url'",
                request=mock_request,
                response=mock_response,
            )

            # Create a response mock with a raise_for_status method that raises our error
            response_mock = MagicMock()
            response_mock.raise_for_status.side_effect = http_error

            # IMPORTANT: Make send an async method
            async def async_send(*args, **kwargs):
                # Verify args are what we expect (if needed)
                self.assertEqual(mock_request, args[0])
                self.assertTrue(kwargs.get("stream"))
                return response_mock

            mock_client.send = async_send

            # Run the test and check for exception
            with self.assertRaises(httpx.HTTPStatusError) as context:
                await customized_model_service.ainfer(
                    messages, model="custom-model", timeout=5.0, stream=True, inference_kwargs={"param": "value"}
                )

            # Verify the exception details
            self.assertEqual(mock_response, context.exception.response)
            self.assertEqual(
                (
                    "Server error: '500 Internal Server Error' for url: 'test_url'\n"
                    "Model Service: test_customized_model_service\n"
                    "Details: test error"
                ),
                str(context.exception),
            )

    async def test_creates_reusable_async_client_lazily(self):
        with patch("httpx.AsyncClient") as mock_client_cls:
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )
            mock_client_cls.assert_not_called()

            self.assertIs(mock_client_cls.return_value, customized_model_service._async_client)
            self.assertIs(mock_client_cls.return_value, customized_model_service._async_client)
            mock_client_cls.assert_called_once_with()

    async def test_aclose_closes_reusable_async_clients(self):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.aclose = AsyncMock()
            customized_model_service = CustomizedHttpxChatModelService(
                name="test_customized_model_service", base_url="test_url"
            )
            mock_client = customized_model_service._async_client

            await customized_model_service.aclose()

            mock_client.aclose.assert_awaited_once_with()


class TestHttpxChatModelServiceRetry(unittest.TestCase):
    """Test retry functionality for _HttpxChatModelService."""

    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Register our custom model service
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_retry_on_timeout_error(self):
        """Test that timeout errors trigger retries."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.Client") as mock_client:
            customized_model_service._client = mock_client.return_value
            # Simulate 2 timeouts, then success
            mock_success_response = Mock()
            mock_success_response.json.return_value = {"messages": [{"role": "assistant", "content": "Success!"}]}
            mock_client.return_value.send.side_effect = [
                httpx.TimeoutException("Request timeout"),
                httpx.TimeoutException("Request timeout"),
                mock_success_response,
            ]

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                replies = customized_model_service.infer(messages, model="custom-model", timeout=5.0, max_retries=3)

                # Verify retry logs (2 warnings for attempts 2 and 3)
                self.assertEqual(2, mock_logger.warning.call_count)

                # Check first retry log contains the right information
                first_call_args = mock_logger.warning.call_args_list[0][0]
                self.assertIn("Retrying", first_call_args[0])
                self.assertEqual("test_customized_model_service", first_call_args[1])
                self.assertEqual("TimeoutException", first_call_args[2])
                self.assertEqual(1, first_call_args[4])  # attempt number
                self.assertEqual(3, first_call_args[5])  # max retries

                # Verify final success
                self.assertEqual(1, len(replies))
                self.assertEqual(ChatMessage.from_assistant("Success!"), replies[0])

                # Verify send was called 3 times (2 failures + 1 success)
                self.assertEqual(3, mock_client.return_value.send.call_count)

    def test_restate_context_skips_tenacity_retry(self):
        """Test that Restate context disables local Tenacity retries."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.Client") as mock_client:
            customized_model_service._client = mock_client.return_value
            mock_client.return_value.send.side_effect = httpx.TimeoutException("Request timeout")

            token = _restate_ctx.set(Mock())
            try:
                with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                    with self.assertRaises(httpx.TimeoutException):
                        customized_model_service.infer(messages, model="custom-model", timeout=5.0, max_retries=3)
            finally:
                _restate_ctx.reset(token)

            mock_logger.warning.assert_not_called()
            self.assertEqual(1, mock_client.return_value.send.call_count)

    def test_retry_on_429_error(self):
        """Test that 429 errors trigger retries."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.Client") as mock_client:
            customized_model_service._client = mock_client.return_value
            # Create mock responses
            mock_429_response = httpx.Response(status_code=429, content=b"Too many requests")
            mock_success_response = Mock()
            mock_success_response.json.return_value = {"messages": [{"role": "assistant", "content": "Success!"}]}

            # Simulate 1 rate limit error, then success
            send_call_count = [0]

            def side_effect_send(*_args, **_kwargs):
                send_call_count[0] += 1
                if send_call_count[0] == 1:
                    mock_response = Mock()
                    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                        "Too many requests",
                        request=Mock(),
                        response=mock_429_response,
                    )
                    return mock_response
                else:
                    return mock_success_response

            mock_client.return_value.send.side_effect = side_effect_send

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                replies = customized_model_service.infer(messages, model="custom-model", timeout=5.0, max_retries=3)

                # Verify retry log
                self.assertEqual(1, mock_logger.warning.call_count)
                self.assertIn("Retrying", str(mock_logger.warning.call_args_list[0]))

                # Verify final success
                self.assertEqual(1, len(replies))
                self.assertEqual(ChatMessage.from_assistant("Success!"), replies[0])

    def test_retry_on_5xx_error(self):
        """Test that configured 5xx errors trigger retries."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service",
            base_url="test_url",
        )

        with patch("httpx.Client") as mock_client:
            customized_model_service._client = mock_client.return_value
            # Create mock responses
            mock_503_response = httpx.Response(status_code=503, content=b"Service unavailable")
            mock_success_response = Mock()
            mock_success_response.json.return_value = {"messages": [{"role": "assistant", "content": "Success!"}]}

            # Simulate 1 503 error, then success
            send_call_count = [0]

            def side_effect_send(*_args, **_kwargs):
                send_call_count[0] += 1
                if send_call_count[0] == 1:
                    mock_response = Mock()
                    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                        "Service unavailable",
                        request=Mock(),
                        response=mock_503_response,
                    )
                    return mock_response
                else:
                    return mock_success_response

            mock_client.return_value.send.side_effect = side_effect_send

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                replies = customized_model_service.infer(messages, model="custom-model", timeout=5.0, max_retries=3)

                # Verify retry log
                self.assertEqual(1, mock_logger.warning.call_count)
                self.assertIn("Retrying", str(mock_logger.warning.call_args_list[0]))

                # Verify final success
                self.assertEqual(1, len(replies))

    def test_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors (e.g., 400, 401) raise immediately without retry."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.Client") as mock_client:
            customized_model_service._client = mock_client.return_value
            # Create a 400 error response
            mock_400_response = httpx.Response(status_code=400, content=b"Bad request")

            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Bad request",
                request=Mock(),
                response=mock_400_response,
            )
            mock_client.return_value.send.return_value = mock_response

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                with self.assertRaises(httpx.HTTPStatusError) as context:
                    customized_model_service.infer(messages, model="custom-model", timeout=5.0)

                # Verify NO retry logs
                self.assertEqual(0, mock_logger.warning.call_count)

                # Verify send was called only once (no retries)
                self.assertEqual(1, mock_client.return_value.send.call_count)

                # Verify the error details
                self.assertIn("Bad request", str(context.exception))

    def test_max_retries_respected(self):
        """Test that max retries limit is respected."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.Client") as mock_client:
            customized_model_service._client = mock_client.return_value
            # Always timeout
            mock_client.return_value.send.side_effect = httpx.TimeoutException("Request timeout")

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                with self.assertRaises(httpx.TimeoutException):
                    customized_model_service.infer(messages, model="custom-model", timeout=5.0, max_retries=2)

                # Verify retry logs (should be max_retries - 1 warnings)
                self.assertEqual(1, mock_logger.warning.call_count)

                # Verify send was called exactly max_retries times
                self.assertEqual(2, mock_client.return_value.send.call_count)

    def test_non_retryable_5xx_codes(self):
        """Test that non-retryable 5xx codes (like 501) do not trigger retries."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service",
            base_url="test_url",
        )

        with patch("httpx.Client") as mock_client:
            customized_model_service._client = mock_client.return_value
            # Create a 501 error response (NOT in retryable list: 500, 502, 503, 504)
            mock_501_response = httpx.Response(status_code=501, content=b"Not implemented")

            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Not implemented",
                request=Mock(),
                response=mock_501_response,
            )
            mock_client.return_value.send.return_value = mock_response

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                with self.assertRaises(httpx.HTTPStatusError):
                    customized_model_service.infer(messages, model="custom-model", timeout=5.0, max_retries=3)

                # Verify NO retry logs (501 is not retryable)
                self.assertEqual(0, mock_logger.warning.call_count)

                # Verify send was called only once
                self.assertEqual(1, mock_client.return_value.send.call_count)


class TestHttpxChatModelServiceRetryAsync(unittest.IsolatedAsyncioTestCase):
    """Test async retry functionality for _HttpxChatModelService."""

    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Register our custom model service
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    async def test_async_retry_on_timeout_error(self):
        """Test that timeout errors trigger retries in async mode."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.AsyncClient") as mock_client_context:
            mock_client = MagicMock()
            mock_client.is_closed = False
            mock_client_context.return_value = mock_client
            customized_model_service._async_client = mock_client

            # Set up request
            mock_request = MagicMock(name="request")
            mock_client.build_request.return_value = mock_request

            # Create success response
            mock_success_response = MagicMock()
            mock_success_response.json.return_value = {"messages": [{"role": "assistant", "content": "Success!"}]}

            # Track send calls
            send_count = [0]

            async def async_send(*_args, **_kwargs):
                send_count[0] += 1
                if send_count[0] <= 2:
                    raise httpx.TimeoutException("Request timeout")
                return mock_success_response

            mock_client.send = async_send

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                replies = await customized_model_service.ainfer(
                    messages, model="custom-model", timeout=5.0, max_retries=3
                )

                # Verify retry logs
                self.assertEqual(2, mock_logger.warning.call_count)
                self.assertIn("Retrying", str(mock_logger.warning.call_args_list[0]))

                # Verify final success
                self.assertEqual(1, len(replies))
                self.assertEqual(ChatMessage.from_assistant("Success!"), replies[0])

                # Verify send was called 3 times
                self.assertEqual(3, send_count[0])

    async def test_async_restate_context_skips_tenacity_retry(self):
        """Test that Restate context disables local async Tenacity retries."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.AsyncClient") as mock_client_context:
            mock_client = MagicMock()
            mock_client.is_closed = False
            mock_client_context.return_value = mock_client
            customized_model_service._async_client = mock_client
            mock_client.build_request.return_value = MagicMock(name="request")

            send_count = [0]

            async def async_send(*_args, **_kwargs):
                send_count[0] += 1
                raise httpx.TimeoutException("Request timeout")

            mock_client.send = async_send

            token = _restate_ctx.set(Mock())
            try:
                with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                    with self.assertRaises(httpx.TimeoutException):
                        await customized_model_service.ainfer(
                            messages, model="custom-model", timeout=5.0, max_retries=3
                        )
            finally:
                _restate_ctx.reset(token)

            mock_logger.warning.assert_not_called()
            self.assertEqual(1, send_count[0])

    async def test_async_retry_on_429_error(self):
        """Test that 429 errors trigger retries in async mode."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.AsyncClient") as mock_client_context:
            mock_client = MagicMock()
            mock_client.is_closed = False
            mock_client_context.return_value = mock_client
            customized_model_service._async_client = mock_client

            # Set up request
            mock_request = MagicMock(name="request")
            mock_client.build_request.return_value = mock_request

            # Create responses
            mock_429_response = httpx.Response(status_code=429, content=b"Too many requests", request=mock_request)
            mock_success_response = MagicMock()
            mock_success_response.json.return_value = {"messages": [{"role": "assistant", "content": "Success!"}]}

            # Track send calls
            send_count = [0]

            async def async_send(*_args, **_kwargs):
                send_count[0] += 1
                if send_count[0] == 1:
                    mock_response = MagicMock()
                    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                        "Too many requests",
                        request=mock_request,
                        response=mock_429_response,
                    )
                    return mock_response
                return mock_success_response

            mock_client.send = async_send

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                replies = await customized_model_service.ainfer(
                    messages, model="custom-model", timeout=5.0, max_retries=3
                )

                # Verify retry log
                self.assertEqual(1, mock_logger.warning.call_count)
                self.assertIn("Retrying", str(mock_logger.warning.call_args_list[0]))

                # Verify final success
                self.assertEqual(1, len(replies))
                self.assertEqual(ChatMessage.from_assistant("Success!"), replies[0])

    async def test_async_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors raise immediately in async mode."""
        messages = [ChatMessage.from_user("Hello world!")]
        customized_model_service = CustomizedHttpxChatModelService(
            name="test_customized_model_service", base_url="test_url"
        )

        with patch("httpx.AsyncClient") as mock_client_context:
            mock_client = MagicMock()
            mock_client.is_closed = False
            mock_client_context.return_value = mock_client
            customized_model_service._async_client = mock_client

            # Set up request
            mock_request = MagicMock(name="request")
            mock_client.build_request.return_value = mock_request

            # Create 400 error
            mock_400_response = httpx.Response(status_code=400, content=b"Bad request", request=mock_request)

            send_count = [0]

            async def async_send(*_args, **_kwargs):
                send_count[0] += 1
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Bad request",
                    request=mock_request,
                    response=mock_400_response,
                )
                return mock_response

            mock_client.send = async_send

            with patch("ravelin.assets.model_services.chat_model_services._base.logger") as mock_logger:
                with self.assertRaises(httpx.HTTPStatusError):
                    await customized_model_service.ainfer(messages, model="custom-model", timeout=5.0)

                # Verify NO retry logs
                self.assertEqual(0, mock_logger.warning.call_count)

                # Verify send was called only once
                self.assertEqual(1, send_count[0])


class TestCollectStreamingChunks(unittest.TestCase):
    """Test _collect_streaming_chunks: emits only GENERATION_DELTA per chunk."""

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)
        self.service = CustomizedHttpxChatModelService(name="test", base_url="test_url")

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_error_propagates_without_emitting_lifecycle(self, mock_get_ctx):
        """Mid-stream error propagates; collector does NOT emit start/error/complete."""
        from ravelin.streaming.event import GENERATION_DELTA, GENERATION_ERROR, GENERATION_START

        mock_ctx = Mock()
        mock_get_ctx.return_value = mock_ctx

        def exploding_iter():
            yield StreamingChunk(content="partial")
            raise RuntimeError("stream broke")

        with self.assertRaises(RuntimeError):
            self.service._collect_streaming_chunks(exploding_iter())

        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit.call_args_list]
        self.assertEqual([GENERATION_DELTA], subtypes)
        self.assertNotIn(GENERATION_START, subtypes)
        self.assertNotIn(GENERATION_ERROR, subtypes)

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_success_emits_only_deltas(self, mock_get_ctx):
        """Successful collection emits one GENERATION_DELTA per chunk and nothing else."""
        from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_DELTA, GENERATION_START

        mock_ctx = Mock()
        mock_get_ctx.return_value = mock_ctx

        chunks = self.service._collect_streaming_chunks(iter([StreamingChunk(content="hi")]))

        self.assertEqual(1, len(chunks))
        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit.call_args_list]
        self.assertEqual([GENERATION_DELTA], subtypes)
        self.assertNotIn(GENERATION_START, subtypes)
        self.assertNotIn(GENERATION_COMPLETE, subtypes)


class TestCollectStreamingChunksAsync(unittest.IsolatedAsyncioTestCase):
    """Test _collect_streaming_chunks_async: emits only GENERATION_DELTA per chunk."""

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)
        self.service = CustomizedHttpxChatModelService(name="test", base_url="test_url")

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    async def test_error_propagates_without_emitting_lifecycle(self, mock_get_ctx):
        from unittest.mock import AsyncMock

        from ravelin.streaming.event import GENERATION_DELTA, GENERATION_ERROR, GENERATION_START

        mock_ctx = Mock()
        mock_ctx.emit_async = AsyncMock()
        mock_get_ctx.return_value = mock_ctx

        async def exploding_iter():
            yield StreamingChunk(content="partial")
            raise RuntimeError("async stream broke")

        with self.assertRaises(RuntimeError):
            await self.service._collect_streaming_chunks_async(exploding_iter())

        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit_async.call_args_list]
        self.assertEqual([GENERATION_DELTA], subtypes)
        self.assertNotIn(GENERATION_START, subtypes)
        self.assertNotIn(GENERATION_ERROR, subtypes)

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    async def test_success_emits_only_deltas(self, mock_get_ctx):
        from unittest.mock import AsyncMock

        from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_DELTA, GENERATION_START

        mock_ctx = Mock()
        mock_ctx.emit_async = AsyncMock()
        mock_get_ctx.return_value = mock_ctx

        async def chunk_iter():
            yield StreamingChunk(content="hi")

        chunks = await self.service._collect_streaming_chunks_async(chunk_iter())

        self.assertEqual(1, len(chunks))
        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit_async.call_args_list]
        self.assertEqual([GENERATION_DELTA], subtypes)
        self.assertNotIn(GENERATION_START, subtypes)
        self.assertNotIn(GENERATION_COMPLETE, subtypes)


class TestInitialInferFailureEmitsGenerationError(unittest.TestCase):
    """The initial request send failing (not mid-stream) still emits generation_error."""

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_infer_initial_failure_emits_generation_error(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_ERROR, EventType

        mock_ctx = Mock()
        mock_get_ctx.return_value = mock_ctx

        mock_response = httpx.Response(status_code=500, content="boom")
        with patch("httpx.Client") as mock_client:
            service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
            mock_client.return_value.build_request.return_value = httpx.Request("POST", "https://example.com")
            mock_client.return_value.send.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error", request=Mock(), response=mock_response
            )
            with self.assertRaises(httpx.HTTPStatusError):
                service.infer([ChatMessage.from_user("hi")], model="m", timeout=5.0, generation_id="g0")

        error_calls = [c for c in mock_ctx.emit.call_args_list if c.kwargs.get("subtype") == GENERATION_ERROR]
        self.assertEqual(1, len(error_calls))
        self.assertEqual(EventType.TOKEN_STREAM, error_calls[0].args[0])
        self.assertEqual("HTTPStatusError", error_calls[0].args[1]["error_type"])
        self.assertEqual("g0", error_calls[0].kwargs.get("generation_id"))


class TestInitialAinferFailureEmitsGenerationError(unittest.IsolatedAsyncioTestCase):
    """Async counterpart: initial request send failing emits generation_error."""

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    async def test_ainfer_initial_failure_emits_generation_error(self, mock_get_ctx):
        from unittest.mock import AsyncMock

        from ravelin.streaming.event import GENERATION_ERROR, EventType

        mock_ctx = Mock()
        mock_ctx.emit_async = AsyncMock()
        mock_get_ctx.return_value = mock_ctx

        mock_response = httpx.Response(status_code=500, content="boom")
        with patch("httpx.AsyncClient") as mock_client_cls:
            service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
            mock_client = MagicMock()
            mock_client.build_request.return_value = httpx.Request("POST", "https://example.com")
            mock_send = AsyncMock(
                side_effect=httpx.HTTPStatusError("Server error", request=Mock(), response=mock_response)
            )
            mock_client.send = mock_send
            mock_client_cls.return_value = mock_client
            service._async_client = mock_client
            with self.assertRaises(httpx.HTTPStatusError):
                await service.ainfer([ChatMessage.from_user("hi")], model="m", timeout=5.0, generation_id="g0")

        error_calls = [c for c in mock_ctx.emit_async.call_args_list if c.kwargs.get("subtype") == GENERATION_ERROR]
        self.assertEqual(1, len(error_calls))
        self.assertEqual(EventType.TOKEN_STREAM, error_calls[0].args[0])
        self.assertEqual("HTTPStatusError", error_calls[0].args[1]["error_type"])
        self.assertEqual("g0", error_calls[0].kwargs.get("generation_id"))


class TestNonStreamingSuccessEmitsGenerationComplete(unittest.TestCase):
    """A successful non-streaming call emits generation_complete so retries self-clean."""

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_infer_success_emits_generation_complete(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_COMPLETE, EventType

        mock_ctx = Mock()
        mock_get_ctx.return_value = mock_ctx

        with patch("httpx.Client") as mock_client:
            service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
            mock_client.return_value.build_request.return_value = httpx.Request("POST", "https://example.com")
            mock_response = Mock()
            mock_response.json.return_value = {"messages": [{"role": "assistant", "content": "hi"}]}
            mock_response.raise_for_status.return_value = None
            mock_client.return_value.send.return_value = mock_response

            service.infer([ChatMessage.from_user("hi")], model="m", timeout=5.0, stream=False, generation_id="g0")

        complete_calls = [c for c in mock_ctx.emit.call_args_list if c.kwargs.get("subtype") == GENERATION_COMPLETE]
        self.assertEqual(1, len(complete_calls))
        self.assertEqual(EventType.TOKEN_STREAM, complete_calls[0].args[0])
        self.assertEqual("g0", complete_calls[0].kwargs.get("generation_id"))


class TestNonStreamingAsyncSuccessEmitsGenerationComplete(unittest.IsolatedAsyncioTestCase):
    """Async counterpart: successful non-streaming ainfer emits generation_complete."""

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    async def test_ainfer_success_emits_generation_complete(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_COMPLETE, EventType

        mock_ctx = Mock()
        mock_ctx.emit_async = AsyncMock()
        mock_get_ctx.return_value = mock_ctx

        with patch("httpx.AsyncClient") as mock_client_cls:
            service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
            mock_client = MagicMock()
            mock_client.build_request.return_value = httpx.Request("POST", "https://example.com")
            mock_response = Mock()
            mock_response.json.return_value = {"messages": [{"role": "assistant", "content": "hi"}]}
            mock_response.raise_for_status.return_value = None
            mock_client.send = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            service._async_client = mock_client

            await service.ainfer(
                [ChatMessage.from_user("hi")], model="m", timeout=5.0, stream=False, generation_id="g0"
            )

        complete_calls = [
            c for c in mock_ctx.emit_async.call_args_list if c.kwargs.get("subtype") == GENERATION_COMPLETE
        ]
        self.assertEqual(1, len(complete_calls))
        self.assertEqual(EventType.TOKEN_STREAM, complete_calls[0].args[0])
        self.assertEqual("g0", complete_calls[0].kwargs.get("generation_id"))


class TestParseResponseFailureEmitsGenerationError(unittest.TestCase):
    """
    A 200 response whose body fails parse_response (e.g. malformed JSON) still
    routes through the single outer try/except in infer and emits generation_error.
    Regression guard for the pre-refactor gap where parse_response failures
    bypassed the request-send try/except entirely.
    """

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_infer_parse_failure_emits_generation_error(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_ERROR

        mock_ctx = Mock()
        mock_get_ctx.return_value = mock_ctx

        service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
        with patch.object(service, "parse_response", side_effect=ValueError("bad json")):
            with patch.object(service._client, "build_request") as mock_build:
                mock_build.return_value = httpx.Request("POST", "https://example.com")
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                with patch.object(service._client, "send", return_value=mock_response):
                    with self.assertRaises(ValueError):
                        service.infer(
                            [ChatMessage.from_user("hi")],
                            model="m",
                            timeout=5.0,
                            stream=False,
                            generation_id="g0",
                        )

        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit.call_args_list]
        self.assertIn(GENERATION_ERROR, subtypes)
        self.assertNotIn(GENERATION_COMPLETE, subtypes)


class TestCompleteEmitFailureSwallowed(unittest.TestCase):
    """
    Streaming emission is best-effort: a backend failure at generation_complete
    must NOT turn a successful LLM call into a caller-visible error, and must
    NOT masquerade as generation_error. Observability degrades for this attempt;
    the caller receives the parsed messages as if streaming were absent.
    """

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_infer_complete_emit_failure_is_swallowed(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_ERROR

        mock_ctx = Mock()

        def emit_side_effect(*args, **kwargs):
            if kwargs.get("subtype") == GENERATION_COMPLETE:
                raise RuntimeError("backend down at complete")

        mock_ctx.emit = Mock(side_effect=emit_side_effect)
        mock_get_ctx.return_value = mock_ctx

        parsed_messages = [ChatMessage.from_assistant(text="hi")]
        service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
        with patch.object(service, "parse_response", return_value=parsed_messages):
            with patch.object(service._client, "build_request") as mock_build:
                mock_build.return_value = httpx.Request("POST", "https://example.com")
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                with patch.object(service._client, "send", return_value=mock_response):
                    result = service.infer(
                        [ChatMessage.from_user("hi")],
                        model="m",
                        timeout=5.0,
                        stream=False,
                        generation_id="g0",
                    )

        # Primary result returned intact.
        self.assertEqual(parsed_messages, result)

        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit.call_args_list]
        # The complete-emit attempt fired (and failed silently).
        self.assertIn(GENERATION_COMPLETE, subtypes)
        # Crucially, no generation_error follows — a failed observability write is
        # NOT the same as a failed inference.
        self.assertNotIn(GENERATION_ERROR, subtypes)


class TestErrorEmitFailurePreservesOriginalException(unittest.TestCase):
    """
    A backend failure while emitting generation_error must not mask the
    original inference exception the caller cares about.
    """

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_infer_error_emit_failure_preserves_original(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_ERROR

        mock_ctx = Mock()

        def emit_side_effect(*args, **kwargs):
            if kwargs.get("subtype") == GENERATION_ERROR:
                raise RuntimeError("backend down at error")

        mock_ctx.emit = Mock(side_effect=emit_side_effect)
        mock_get_ctx.return_value = mock_ctx

        service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
        # parse_response fails — the "real" inference error.
        with patch.object(service, "parse_response", side_effect=ValueError("bad json")):
            with patch.object(service._client, "build_request") as mock_build:
                mock_build.return_value = httpx.Request("POST", "https://example.com")
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                with patch.object(service._client, "send", return_value=mock_response):
                    # Caller sees the original ValueError, not the emit RuntimeError.
                    with self.assertRaises(ValueError) as cm:
                        service.infer(
                            [ChatMessage.from_user("hi")],
                            model="m",
                            timeout=5.0,
                            stream=False,
                            generation_id="g0",
                        )

        self.assertEqual("bad json", str(cm.exception))


class TestAparseResponseFailureEmitsGenerationError(unittest.IsolatedAsyncioTestCase):
    """Async counterpart: aparse_response failure routes through the outer try/except."""

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    async def test_ainfer_parse_failure_emits_generation_error(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_ERROR

        mock_ctx = Mock()
        mock_ctx.emit_async = AsyncMock()
        mock_get_ctx.return_value = mock_ctx

        service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
        with patch.object(service, "aparse_response", new=AsyncMock(side_effect=ValueError("bad json"))):
            mock_client = MagicMock()
            mock_client.build_request.return_value = httpx.Request("POST", "https://example.com")
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.send = AsyncMock(return_value=mock_response)
            service._async_client = mock_client
            with self.assertRaises(ValueError):
                await service.ainfer(
                    [ChatMessage.from_user("hi")],
                    model="m",
                    timeout=5.0,
                    stream=False,
                    generation_id="g0",
                )

        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit_async.call_args_list]
        self.assertIn(GENERATION_ERROR, subtypes)
        self.assertNotIn(GENERATION_COMPLETE, subtypes)


class TestCompleteEmitFailureSwallowedAsync(unittest.IsolatedAsyncioTestCase):
    """Async counterpart of TestCompleteEmitFailureSwallowed."""

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    async def test_ainfer_complete_emit_failure_is_swallowed(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_ERROR

        async def emit_side_effect(*args, **kwargs):
            if kwargs.get("subtype") == GENERATION_COMPLETE:
                raise RuntimeError("backend down at complete")

        mock_ctx = Mock()
        mock_ctx.emit_async = AsyncMock(side_effect=emit_side_effect)
        mock_get_ctx.return_value = mock_ctx

        parsed_messages = [ChatMessage.from_assistant(text="hi")]
        service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
        with patch.object(service, "aparse_response", new=AsyncMock(return_value=parsed_messages)):
            mock_client = MagicMock()
            mock_client.build_request.return_value = httpx.Request("POST", "https://example.com")
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.send = AsyncMock(return_value=mock_response)
            service._async_client = mock_client
            result = await service.ainfer(
                [ChatMessage.from_user("hi")],
                model="m",
                timeout=5.0,
                stream=False,
                generation_id="g0",
            )

        self.assertEqual(parsed_messages, result)
        subtypes = [c.kwargs.get("subtype") for c in mock_ctx.emit_async.call_args_list]
        self.assertIn(GENERATION_COMPLETE, subtypes)
        self.assertNotIn(GENERATION_ERROR, subtypes)


class TestErrorEmitFailurePreservesOriginalExceptionAsync(unittest.IsolatedAsyncioTestCase):
    """Async counterpart of TestErrorEmitFailurePreservesOriginalException."""

    def setUp(self):
        self.registry = {}
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.registry_patcher.start()
        register_model_service("custom_chat_format")(CustomizedHttpxChatModelService)

    def tearDown(self):
        self.registry_patcher.stop()

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    async def test_ainfer_error_emit_failure_preserves_original(self, mock_get_ctx):
        from ravelin.streaming.event import GENERATION_ERROR

        async def emit_side_effect(*args, **kwargs):
            if kwargs.get("subtype") == GENERATION_ERROR:
                raise RuntimeError("backend down at error")

        mock_ctx = Mock()
        mock_ctx.emit_async = AsyncMock(side_effect=emit_side_effect)
        mock_get_ctx.return_value = mock_ctx

        service = CustomizedHttpxChatModelService(name="svc", base_url="test_url")
        with patch.object(service, "aparse_response", new=AsyncMock(side_effect=ValueError("bad json"))):
            mock_client = MagicMock()
            mock_client.build_request.return_value = httpx.Request("POST", "https://example.com")
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.send = AsyncMock(return_value=mock_response)
            service._async_client = mock_client
            with self.assertRaises(ValueError) as cm:
                await service.ainfer(
                    [ChatMessage.from_user("hi")],
                    model="m",
                    timeout=5.0,
                    stream=False,
                    generation_id="g0",
                )

        self.assertEqual("bad json", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
