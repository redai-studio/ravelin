import unittest
from typing import Optional, Dict, Any
from unittest.mock import Mock, patch

import httpx
from haystack import Document
from haystack.utils import Secret

from ravelin.assets.tool_services import register_tool_service
from ravelin.assets.tool_services.websearch_services._base import _HttpxWebSearchService


class TestCustomizedWebSearchService(unittest.TestCase):
    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.tool_services._base.TOOL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Register our custom tool service
        register_tool_service("custom_search_format")(self.CustomizedHttpxWebSearchService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    @register_tool_service("custom_search_format")
    class CustomizedHttpxWebSearchService(_HttpxWebSearchService):
        def build_request(
            self,
            client: httpx.Client,
            query: str,
            *,
            num: Optional[int] = None,
            page: Optional[int] = None,
            search_kwargs: Optional[Dict[str, Any]] = None,
        ) -> httpx.Request:
            search_kwargs = search_kwargs or {}
            return client.build_request(
                "GET",
                url=self.uri,
                params={
                    "q": query,
                    "num": num,
                    "page": page,
                    **search_kwargs,
                },
            )

        def parse_response(self, response: httpx.Response) -> list[Document]:
            results = response.json().get("results", [])
            return [
                Document(
                    content=result["content"],
                    meta={
                        "position": idx,
                        "title": result["title"],
                        "snippet": result["snippet"],
                        "url": result["url"],
                    },
                )
                for idx, result in enumerate(results)
            ]

    def test_initialization(self):
        search_service = self.CustomizedHttpxWebSearchService(
            name="test_search_service",
            uri="https://api.search.test",
            api_key=Secret.from_token("test_key"),
            client_kwargs={"headers": {"Custom-Header": "value"}},
        )

        self.assertEqual("test_search_service", search_service.name)
        self.assertEqual(
            {
                "uri": "https://api.search.test",
                "organization": None,
                "client_kwargs": {"headers": {"Custom-Header": "value"}},
                "call_api_format": "custom_search_format",
            },
            search_service.configs,
        )
        self.assertEqual("test_key", search_service.secrets["api_key"].resolve_value())

    def test_search(self):
        search_service = self.CustomizedHttpxWebSearchService(
            name="test_search_service",
            uri="https://api.search.test",
        )

        mock_results = {
            "results": [
                {
                    "title": "Test Page",
                    "content": "Test content",
                    "snippet": "Test snippet",
                    "url": "https://test.com",
                }
            ]
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.send.return_value.raise_for_status.return_value.json.return_value = mock_results

            results = search_service(
                query="test query",
                num=10,
                page=1,
                timeout=5.0,
                search_kwargs={"filter": "news"},
            )

            # Verify client initialization
            mock_client.assert_called_once_with()
            # Check that the timeout was applied to the request
            mock_client.return_value.build_request.return_value.extensions.__setitem__.assert_called_once_with(
                "timeout", httpx.Timeout(5.0).as_dict()
            )

            # Verify request building
            mock_client.return_value.build_request.assert_called_once_with(
                "GET",
                url="https://api.search.test",
                params={
                    "q": "test query",
                    "num": 10,
                    "page": 1,
                    "filter": "news",
                },
            )

            # Verify response handling
            self.assertEqual(1, len(results))
            self.assertIsInstance(results[0], Document)
            self.assertEqual("Test content", results[0].content)
            self.assertEqual(
                {
                    "position": 0,
                    "title": "Test Page",
                    "snippet": "Test snippet",
                    "url": "https://test.com",
                },
                results[0].meta,
            )

    def test_search_error(self):
        search_service = self.CustomizedHttpxWebSearchService(
            name="test_search_service",
            uri="https://api.search.test",
        )

        mock_response = httpx.Response(status_code=500, content="search error")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.send.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error: '500 Internal Server Error' for url: 'https://api.search.test'",
                request=Mock(),
                response=mock_response,
            )

            with self.assertRaises(httpx.HTTPStatusError) as context:
                search_service(query="test query")

            self.assertEqual(mock_response, context.exception.response)
            self.assertEqual(
                (
                    "Server error: '500 Internal Server Error' for url: 'https://api.search.test'\n"
                    "WebSearch Service: test_search_service\n"
                    "Details: search error"
                ),
                str(context.exception),
            )

    def test_client_kwargs_property(self):
        search_service = self.CustomizedHttpxWebSearchService(
            name="test_scraper_service",
            uri="https://api.scraper.test",
            client_kwargs={"headers": {"Custom-Header": "value"}},
        )

        self.assertEqual({"headers": {"Custom-Header": "value"}}, search_service.client_kwargs)

        # Test that modifying the property updates configs
        search_service.configs["client_kwargs"] = {"timeout": 30}
        self.assertEqual({"timeout": 30}, search_service.client_kwargs)


if __name__ == "__main__":
    unittest.main()
