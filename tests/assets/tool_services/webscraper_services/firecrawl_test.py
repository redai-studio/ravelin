import json
import unittest
from unittest.mock import patch

import httpx
from haystack import Document
from haystack.utils import Secret

from ravelin.assets.tool_services.webscraper_services.firecrawl import (
    FirecrawlWebScraperService,
    FIRECRAWL_BASE_URI,
    FIRECRAWL_SCRAPE_ENDPOINT_SUFFIX,
)


class TestFirecrawlWebScraperService(unittest.TestCase):
    def setUp(self):
        self.webscraper_service = FirecrawlWebScraperService(
            name="test_firecrawl_webscraper_service",
            api_key=Secret.from_token("test_key"),
            organization="test_org",
        )
        self.client = httpx.Client()

    def test_initialization(self):
        self.assertEqual("test_firecrawl_webscraper_service", self.webscraper_service.name)
        self.assertEqual(
            {
                "uri": FIRECRAWL_BASE_URI,
                "organization": "test_org",
                "call_api_format": "firecrawl-webscraper",
            },
            self.webscraper_service.configs,
        )
        self.assertEqual("test_key", self.webscraper_service.secrets["api_key"].resolve_value())
        expected_endpoint = f"{FIRECRAWL_BASE_URI.rstrip('/')}/{FIRECRAWL_SCRAPE_ENDPOINT_SUFFIX}"
        self.assertEqual(expected_endpoint, self.webscraper_service.scrape_endpoint)

    def test_build_request(self):
        request = self.webscraper_service.build_request(
            self.client,
            "https://example.com",
            timeout=10.0,
            output_formats=["markdown", "html"],
            scraper_kwargs={"javascript": True},
        )

        self.assertEqual("POST", request.method)
        expected_endpoint = f"{FIRECRAWL_BASE_URI.rstrip('/')}/{FIRECRAWL_SCRAPE_ENDPOINT_SUFFIX}"
        self.assertEqual(expected_endpoint, str(request.url))

        # Check request headers
        self.assertEqual("Bearer test_key", request.headers.get("Authorization"))
        self.assertEqual("application/json", request.headers.get("Content-Type"))

        # Check request body
        body = json.loads(request.content)
        self.assertEqual("https://example.com", body["url"])
        self.assertEqual(10000, body["timeout"])  # Check timeout is converted to ms
        self.assertEqual(["markdown", "html"], body["formats"])
        self.assertTrue(body["javascript"])

    def test_parse_response(self):
        response_body = {
            "data": {
                "markdown": "# Example Website\n\nThis is an example website content.",
                "html": "<h1>Example Website</h1><p>This is an example website content.</p>",
                "metadata": {
                    "title": "Example Website",
                    "sourceURL": "https://example.com",
                    "statusCode": 200,
                    "description": "An example website for testing",
                    "error": None,
                },
            }
        }

        response = httpx.Response(
            status_code=200,
            content=json.dumps(response_body).encode(),
        )

        documents = self.webscraper_service.parse_response(response, output_formats=["markdown", "html"])

        self.assertEqual(1, len(documents))
        self.assertIsInstance(documents[0], Document)

        # The first format in the list (markdown) should be used as the main content
        self.assertEqual("# Example Website\n\nThis is an example website content.", documents[0].content)

        # Check metadata
        self.assertEqual("markdown", documents[0].meta["content_type"])
        self.assertEqual(200, documents[0].meta["status_code"])
        self.assertEqual("https://example.com", documents[0].meta["url"])
        self.assertEqual("Example Website", documents[0].meta["title"])
        self.assertEqual("An example website for testing", documents[0].meta["description"])
        self.assertEqual(
            "<h1>Example Website</h1><p>This is an example website content.</p>", documents[0].meta["html"]
        )

    def test_parse_response_with_default_format(self):
        response_body = {
            "data": {
                "markdown": "# Example Website\n\nThis is an example website content.",
                "metadata": {
                    "title": "Example Website",
                    "sourceURL": "https://example.com",
                    "statusCode": 200,
                    "description": "An example website for testing",
                },
            }
        }

        response = httpx.Response(
            status_code=200,
            content=json.dumps(response_body).encode(),
        )

        # No output_formats passed, should default to ["markdown"]
        documents = self.webscraper_service.parse_response(response)

        self.assertEqual(1, len(documents))
        self.assertEqual("# Example Website\n\nThis is an example website content.", documents[0].content)
        self.assertEqual("markdown", documents[0].meta["content_type"])

    def test_parse_response_with_error(self):
        response_body = {
            "data": {
                "markdown": "",
                "metadata": {
                    "title": "",
                    "sourceURL": "https://invalid-url.example",
                    "statusCode": 404,
                    "description": "",
                    "error": "Page not found",
                },
            }
        }

        response = httpx.Response(
            status_code=200,
            content=json.dumps(response_body).encode(),
        )

        documents = self.webscraper_service.parse_response(response)

        self.assertEqual(1, len(documents))
        self.assertEqual(None, documents[0].content)
        self.assertEqual("Page not found", documents[0].meta["error"])
        self.assertEqual(404, documents[0].meta["status_code"])

    def test_call_method(self):
        with patch.object(FirecrawlWebScraperService, "__call__", return_value=[Document(content="test")]) as mock_call:
            result = self.webscraper_service(
                url="https://example.com", timeout=5.0, output_formats=["markdown"], scraper_kwargs={"javascript": True}
            )

            mock_call.assert_called_once_with(
                url="https://example.com",
                timeout=5.0,
                output_formats=["markdown"],
                scraper_kwargs={"javascript": True},
            )
            self.assertEqual([Document(content="test")], result)


if __name__ == "__main__":
    unittest.main()
