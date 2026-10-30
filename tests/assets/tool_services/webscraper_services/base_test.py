import unittest
from typing import Optional, Dict, Any, List
from unittest.mock import Mock, patch

import httpx
from haystack import Document
from haystack.utils import Secret

from ravelin.assets.tool_services import register_tool_service
from ravelin.assets.tool_services.webscraper_services._base import _HttpxWebScraperService


class TestCustomizedWebScraperService(unittest.TestCase):
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
        register_tool_service("custom_scraper_format")(self.CustomizedHttpxWebScraperService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    @register_tool_service("custom_scraper_format")
    class CustomizedHttpxWebScraperService(_HttpxWebScraperService):
        def build_request(
            self,
            client: httpx.Client,
            url: str,
            *,
            timeout: Optional[float] = None,
            output_formats: Optional[List[str]] = None,
            scraper_kwargs: Optional[Dict[str, Any]] = None,
        ) -> httpx.Request:
            scraper_kwargs = scraper_kwargs or {}
            params = {
                "url": url,
                **scraper_kwargs,
            }
            if timeout:
                params["timeout"] = timeout
            if output_formats:
                params["formats"] = ",".join(output_formats)

            return client.build_request("GET", url=self.uri, params=params)

        def parse_response(
            self, response: httpx.Response, output_formats: Optional[List[str]] = None
        ) -> List[Document]:
            data = response.json()
            return [
                Document(
                    content=data["content"],
                    meta={
                        "title": data["title"],
                        "url": data["url"],
                        "content_type": "text",
                        "status_code": data["status_code"],
                    },
                )
            ]

    def test_initialization(self):
        scraper_service = self.CustomizedHttpxWebScraperService(
            name="test_scraper_service",
            uri="https://api.scraper.test",
            api_key=Secret.from_token("test_key"),
            client_kwargs={"headers": {"Custom-Header": "value"}},
        )

        self.assertEqual("test_scraper_service", scraper_service.name)
        self.assertEqual(
            {
                "uri": "https://api.scraper.test",
                "organization": None,
                "client_kwargs": {"headers": {"Custom-Header": "value"}},
                "call_api_format": "custom_scraper_format",
            },
            scraper_service.configs,
        )
        self.assertEqual("test_key", scraper_service.secrets["api_key"].resolve_value())

    def test_scrape(self):
        scraper_service = self.CustomizedHttpxWebScraperService(
            name="test_scraper_service",
            uri="https://api.scraper.test",
        )

        mock_response_data = {
            "content": "Scraped content of the website",
            "title": "Test Page",
            "url": "https://test.com",
            "status_code": 200,
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.send.return_value.raise_for_status.return_value.json.return_value = (
                mock_response_data
            )

            results = scraper_service(
                url="https://test.com",
                timeout=10.0,
                output_formats=["text", "markdown"],
                scraper_kwargs={"javascript": True},
            )

            # Verify client initialization
            mock_client.assert_called_once_with()
            # Check that the timeout was applied to the request
            mock_client.return_value.build_request.return_value.extensions.__setitem__.assert_called_once_with(
                "timeout", httpx.Timeout(10.0).as_dict()
            )

            # Verify request building
            mock_client.return_value.build_request.assert_called_once_with(
                "GET",
                url="https://api.scraper.test",
                params={
                    "url": "https://test.com",
                    "timeout": 10.0,
                    "formats": "text,markdown",
                    "javascript": True,
                },
            )

            # Verify response handling
            self.assertEqual(1, len(results))
            self.assertIsInstance(results[0], Document)
            self.assertEqual("Scraped content of the website", results[0].content)
            self.assertEqual(
                {
                    "title": "Test Page",
                    "url": "https://test.com",
                    "content_type": "text",
                    "status_code": 200,
                },
                results[0].meta,
            )

    def test_scrape_error(self):
        scraper_service = self.CustomizedHttpxWebScraperService(
            name="test_scraper_service",
            uri="https://api.scraper.test",
        )

        mock_response = httpx.Response(status_code=500, content="scraper error")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.send.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server error: '500 Internal Server Error' for url: 'https://api.scraper.test'",
                request=Mock(),
                response=mock_response,
            )

            with self.assertRaises(httpx.HTTPStatusError) as context:
                scraper_service(url="https://test.com")

            self.assertEqual(mock_response, context.exception.response)
            self.assertEqual(
                (
                    "Server error: '500 Internal Server Error' for url: 'https://api.scraper.test'\n"
                    "WebScrapper Service: test_scraper_service\n"
                    "Details: scraper error"
                ),
                str(context.exception),
            )

    def test_client_kwargs_property(self):
        scraper_service = self.CustomizedHttpxWebScraperService(
            name="test_scraper_service",
            uri="https://api.scraper.test",
            client_kwargs={"headers": {"Custom-Header": "value"}},
        )

        self.assertEqual({"headers": {"Custom-Header": "value"}}, scraper_service.client_kwargs)

        # Test that modifying the property updates configs
        scraper_service.configs["client_kwargs"] = {"timeout": 30}
        self.assertEqual({"timeout": 30}, scraper_service.client_kwargs)


if __name__ == "__main__":
    unittest.main()
