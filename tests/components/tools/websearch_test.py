import unittest
from unittest.mock import Mock, patch, call

from haystack.dataclasses import Document

from ravelin.assets.tool_services import ToolService, WebSearchService
from ravelin.components.tools.websearch import WebSearch
from ravelin.streaming import EventType
from ravelin.dataclasses.asset_reference import AssetReference


class TestWebsearch(unittest.TestCase):
    def setUp(self):
        self.search_results = [Document(content="search result", meta={"url": "https://example.com"})]
        self.websearch_service = Mock(spec=WebSearchService, return_value=self.search_results)
        self.websearch_service.reference = AssetReference(name="test-websearch-service", source="local")

        self.websearch = WebSearch(
            websearch_service=self.websearch_service,
            num=10,
            page=1,
            timeout=20.0,
            search_kwargs={"foo": "bar"},
        )

    def test_initialization(self):
        self.assertEqual(self.websearch_service, self.websearch.websearch_service)
        self.assertEqual(10, self.websearch.num)
        self.assertEqual(1, self.websearch.page)
        self.assertEqual(20.0, self.websearch.timeout)
        self.assertEqual({"foo": "bar"}, self.websearch.search_kwargs)

    def test_to_dict(self):
        websearch_dict = self.websearch.to_dict()

        self.assertIn("init_parameters", websearch_dict)
        self.assertIn("type", websearch_dict)
        self.assertEqual("ravelin.components.tools.websearch.WebSearch", websearch_dict["type"])
        self.assertEqual(
            self.websearch_service.reference.to_dict(), websearch_dict["init_parameters"]["websearch_service"]
        )
        self.assertEqual(10, websearch_dict["init_parameters"]["num"])
        self.assertEqual(1, websearch_dict["init_parameters"]["page"])
        self.assertEqual(20.0, websearch_dict["init_parameters"]["timeout"])
        self.assertEqual({"foo": "bar"}, websearch_dict["init_parameters"]["search_kwargs"])

    @patch("ravelin.components.tools.websearch.ToolServiceFactory.load_from_reference")
    def test_from_dict_happy_path(self, mock_load_service_from_reference):
        data = {
            "type": "ravelin.components.tools.websearch.WebSearch",
            "init_parameters": {
                "websearch_service": {
                    "name": "test-websearch-service",
                    "source": "local",
                },
                "num": 10,
                "page": 1,
                "timeout": 5.0,
                "search_kwargs": {"foo": "bar"},
            },
        }
        mock_load_service_from_reference.return_value = self.websearch_service
        websearch = WebSearch.from_dict(data)

        mock_load_service_from_reference.assert_called_once_with(
            AssetReference.from_dict(data["init_parameters"]["websearch_service"])
        )
        self.assertIsInstance(websearch, WebSearch)
        self.assertEqual(self.websearch_service, websearch.websearch_service)
        self.assertEqual(10, websearch.num)
        self.assertEqual(1, websearch.page)
        self.assertEqual(5.0, websearch.timeout)
        self.assertEqual({"foo": "bar"}, websearch.search_kwargs)

    @patch("ravelin.components.tools.websearch.ToolServiceFactory.load_from_reference")
    def test_from_dict_with_invalid_model_service_type(self, mock_load_service_from_reference):
        data = {
            "type": "ravelin.components.tools.websearch.WebSearch",
            "init_parameters": {
                "websearch_service": {
                    "name": "test-websearch-service",
                    "source": "local",
                },
                "num": 10,
                "page": 1,
                "timeout": 5.0,
                "search_kwargs": {"foo": "bar"},
            },
        }

        mock_load_service_from_reference.return_value = Mock(spec=ToolService)
        with self.assertRaisesRegex(
            TypeError, "The websearch service must be an instance of WebSearchService, got <class 'unittest.mock.Mock'>"
        ):
            WebSearch.from_dict(data)

    @patch("ravelin.components.tools.websearch.get_component_context")
    def test_run_with_successful_response(self, mock_get_comp_ctx):
        mock_comp_ctx = Mock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        result = self.websearch.run(query="test query")

        self.assertEqual(
            call("test query", num=10, page=1, timeout=20.0, search_kwargs={"foo": "bar"}),
            self.websearch_service.call_args,
        )
        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))
        self.assertEqual("search result", result["documents"][0].content)

        mock_comp_ctx.emit.assert_any_call(EventType.COMPONENT_ACTIVITY, "test query", subtype="search_query")
        mock_comp_ctx.emit.assert_any_call(EventType.COMPONENT_ACTIVITY, self.search_results, subtype="search_results")

    @patch("ravelin.components.tools.websearch.get_component_context")
    def test_run_without_streaming_session(self, mock_get_comp_ctx):
        mock_get_comp_ctx.return_value = None

        result = self.websearch.run(query="test query")

        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))

    def test_run_with_unsuccessful_response(self):
        self.websearch_service.side_effect = ValueError("foobar")
        with self.assertRaisesRegex(ValueError, "foobar"):
            self.websearch.run("test query")


class TestWebsearchAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.search_results = [Document(content="search result", meta={"url": "https://example.com"})]
        self.websearch_service = Mock(spec=WebSearchService, return_value=self.search_results)
        self.websearch_service.reference = AssetReference(name="test-websearch-service", source="local")
        self.websearch = WebSearch(
            websearch_service=self.websearch_service,
            num=10,
            page=1,
            timeout=20.0,
            search_kwargs={"foo": "bar"},
        )

    @patch("ravelin.components.tools.websearch.get_component_context")
    async def test_run_async_emits_events(self, mock_get_comp_ctx):
        from unittest.mock import AsyncMock

        mock_comp_ctx = Mock()
        mock_comp_ctx.emit_async = AsyncMock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        result = await self.websearch.run_async(query="test query")

        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))

        mock_comp_ctx.emit_async.assert_any_call(EventType.COMPONENT_ACTIVITY, "test query", subtype="search_query")
        mock_comp_ctx.emit_async.assert_any_call(
            EventType.COMPONENT_ACTIVITY, self.search_results, subtype="search_results"
        )

    @patch("ravelin.components.tools.websearch.get_component_context")
    async def test_run_async_without_streaming_session(self, mock_get_comp_ctx):
        mock_get_comp_ctx.return_value = None

        result = await self.websearch.run_async(query="test query")

        self.assertIn("documents", result)
        self.assertEqual(1, len(result["documents"]))


if __name__ == "__main__":
    unittest.main()
