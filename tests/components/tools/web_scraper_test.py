# tests/components/tools/web_scraper_test.py
import unittest
from unittest.mock import Mock, patch, call

from haystack.dataclasses import Document

from ravelin.assets.tool_services import ToolService, WebScraperService
from ravelin.components.tools.web_scraper import WebScraper
from ravelin.streaming import EventType
from ravelin.dataclasses.asset_reference import AssetReference


class TestWebScraper(unittest.TestCase):
    def setUp(self):
        self.scrape_results = [
            Document(content="scraped content", meta={"url": "https://example.com", "title": "Example"})
        ]
        self.webscraper_service = Mock(spec=WebScraperService, return_value=self.scrape_results)
        self.webscraper_service.reference = AssetReference(name="test-webscraper-service", source="local")

        self.webscraper = WebScraper(
            webscraper_service=self.webscraper_service,
            timeout=30.0,
            output_formats=["markdown", "html"],
            scraper_kwargs={"javascript": True},
        )

    def test_initialization(self):
        self.assertEqual(self.webscraper_service, self.webscraper.webscraper_service)
        self.assertEqual(30.0, self.webscraper.timeout)
        self.assertEqual(["markdown", "html"], self.webscraper.output_formats)
        self.assertEqual({"javascript": True}, self.webscraper.scraper_kwargs)

    def test_to_dict(self):
        webscraper_dict = self.webscraper.to_dict()

        self.assertIn("init_parameters", webscraper_dict)
        self.assertIn("type", webscraper_dict)
        self.assertEqual("ravelin.components.tools.web_scraper.WebScraper", webscraper_dict["type"])
        self.assertEqual(
            self.webscraper_service.reference.to_dict(), webscraper_dict["init_parameters"]["webscraper_service"]
        )
        self.assertEqual(30.0, webscraper_dict["init_parameters"]["timeout"])
        self.assertEqual(["markdown", "html"], webscraper_dict["init_parameters"]["output_formats"])
        self.assertEqual({"javascript": True}, webscraper_dict["init_parameters"]["scraper_kwargs"])

    @patch("ravelin.components.tools.web_scraper.ToolServiceFactory.load_from_reference")
    def test_from_dict_happy_path(self, mock_load_service_from_reference):
        data = {
            "type": "ravelin.components.tools.web_scraper.WebScraper",
            "init_parameters": {
                "webscraper_service": {
                    "name": "test-webscraper-service",
                    "source": "local",
                },
                "timeout": 15.0,
                "output_formats": ["markdown", "text"],
                "scraper_kwargs": {"javascript": True},
            },
        }
        mock_load_service_from_reference.return_value = self.webscraper_service
        webscraper = WebScraper.from_dict(data)

        mock_load_service_from_reference.assert_called_once_with(
            AssetReference.from_dict(data["init_parameters"]["webscraper_service"])
        )
        self.assertIsInstance(webscraper, WebScraper)
        self.assertEqual(self.webscraper_service, webscraper.webscraper_service)
        self.assertEqual(15.0, webscraper.timeout)
        self.assertEqual(["markdown", "text"], webscraper.output_formats)
        self.assertEqual({"javascript": True}, webscraper.scraper_kwargs)

    @patch("ravelin.components.tools.web_scraper.ToolServiceFactory.load_from_reference")
    def test_from_dict_with_invalid_service_type(self, mock_load_service_from_reference):
        data = {
            "type": "ravelin.components.tools.web_scraper.WebScraper",
            "init_parameters": {
                "webscraper_service": {
                    "name": "test-webscraper-service",
                    "source": "local",
                },
                "timeout": 15.0,
                "output_formats": ["markdown"],
                "scraper_kwargs": {},
            },
        }

        mock_load_service_from_reference.return_value = Mock(spec=ToolService)
        with self.assertRaisesRegex(
            TypeError,
            "The webscraper service must be an instance of WebScraperService, got <class 'unittest.mock.Mock'>",
        ):
            WebScraper.from_dict(data)

    @patch("ravelin.components.tools.web_scraper.get_component_context")
    def test_run_with_successful_response(self, mock_get_comp_ctx):
        mock_comp_ctx = Mock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        result = self.webscraper.run(url="https://example.com")

        self.assertEqual(
            call(
                "https://example.com",
                timeout=30.0,
                output_formats=["markdown", "html"],
                scraper_kwargs={"javascript": True},
            ),
            self.webscraper_service.call_args,
        )
        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))
        self.assertEqual("scraped content", result["documents"][0].content)

        mock_comp_ctx.emit.assert_any_call(
            EventType.COMPONENT_ACTIVITY, "https://example.com", subtype="scrape_webpage"
        )
        mock_comp_ctx.emit.assert_any_call(EventType.COMPONENT_ACTIVITY, self.scrape_results, subtype="scrape_results")

    @patch("ravelin.components.tools.web_scraper.get_component_context")
    def test_run_without_streaming_session(self, mock_get_comp_ctx):
        mock_get_comp_ctx.return_value = None

        result = self.webscraper.run(url="https://example.com")

        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))

    def test_run_with_additional_scraper_kwargs(self):
        self.webscraper.run(url="https://example.com", scraper_kwargs={"follow_links": True})

        self.assertEqual(
            call(
                "https://example.com",
                timeout=30.0,
                output_formats=["markdown", "html"],
                scraper_kwargs={"javascript": True, "follow_links": True},
            ),
            self.webscraper_service.call_args,
        )

    def test_run_with_unsuccessful_response(self):
        self.webscraper_service.side_effect = ValueError("Failed to scrape webpage")
        with self.assertRaisesRegex(ValueError, "Failed to scrape webpage"):
            self.webscraper.run("https://example.com")


class TestWebScraperAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scrape_results = [
            Document(content="scraped content", meta={"url": "https://example.com", "title": "Example"})
        ]
        self.webscraper_service = Mock(spec=WebScraperService, return_value=self.scrape_results)
        self.webscraper_service.reference = AssetReference(name="test-webscraper-service", source="local")
        self.webscraper = WebScraper(
            webscraper_service=self.webscraper_service,
            timeout=30.0,
            output_formats=["markdown", "html"],
            scraper_kwargs={"javascript": True},
        )

    @patch("ravelin.components.tools.web_scraper.get_component_context")
    async def test_run_async_emits_events(self, mock_get_comp_ctx):
        from unittest.mock import AsyncMock

        mock_comp_ctx = Mock()
        mock_comp_ctx.emit_async = AsyncMock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        result = await self.webscraper.run_async(url="https://example.com")

        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))

        mock_comp_ctx.emit_async.assert_any_call(
            EventType.COMPONENT_ACTIVITY, "https://example.com", subtype="scrape_webpage"
        )
        mock_comp_ctx.emit_async.assert_any_call(
            EventType.COMPONENT_ACTIVITY, self.scrape_results, subtype="scrape_results"
        )

    @patch("ravelin.components.tools.web_scraper.get_component_context")
    async def test_run_async_without_streaming_session(self, mock_get_comp_ctx):
        mock_get_comp_ctx.return_value = None

        result = await self.webscraper.run_async(url="https://example.com")

        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))


if __name__ == "__main__":
    unittest.main()
