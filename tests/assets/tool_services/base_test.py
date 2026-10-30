import os
import unittest
from typing import Any, Optional
from unittest.mock import patch

from haystack.tools import Tool
from haystack.utils import Secret

from ravelin.assets._base import resolve_secret
from ravelin.assets.tool_services._base import (
    _ToolService,
    register_tool_service,
    TOOL_SERVICE_REGISTRY,
)


# Define mock classes without decorators to avoid registering them globally
class MockWebScraperService(_ToolService):
    def __call__(self, *args, **kwargs) -> Any:
        pass


class MockWebSearchService(_ToolService):
    def __call__(self, *args, **kwargs) -> Any:
        pass


class TestToolService(unittest.TestCase):
    """Tests for tool service functionality"""

    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.tool_services._base.TOOL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Now register our mock classes using the decorator function
        register_tool_service("webscraper")(MockWebScraperService)
        register_tool_service("websearch")(MockWebSearchService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_registry_access(self):
        """Test basic registry functionality"""

        # Verify we can access the pre-registered mock classes
        registry = TOOL_SERVICE_REGISTRY.get_all()
        self.assertIsInstance(registry, dict)
        self.assertEqual(2, len(registry))
        self.assertTrue(TOOL_SERVICE_REGISTRY.contains("webscraper"))
        self.assertTrue(TOOL_SERVICE_REGISTRY.contains("websearch"))

    def test_initialization(self):
        """Test tool service initialization with all parameters"""

        @register_tool_service("simple_format")
        class SimpleTestToolService(_ToolService):
            """A simple tool service for testing."""

            def __call__(self, query: str, num_results: Optional[int] = 5) -> list[str]:
                """
                A simple search function.

                :param query: The search query
                :param num_results: Number of results to return
                :return: List of results
                """
                # Handle the None case explicitly to fix type checking
                count = 5 if num_results is None else num_results
                return [f"Result {i} for {query}" for i in range(count)]

        tool_service = SimpleTestToolService(
            name="test_tool", uri="https://api.test.com", api_key=Secret.from_token("test_key"), organization="test_org"
        )

        self.assertEqual("test_tool", tool_service.name)
        self.assertEqual(
            {
                "uri": "https://api.test.com",
                "organization": "test_org",
                "call_api_format": "simple_format",
            },
            tool_service.configs,
        )
        self.assertEqual("test_key", tool_service.secrets["api_key"].resolve_value())

    def test_api_key_property(self):
        """Test api_key property getter and setter"""

        @register_tool_service("simple_format")
        class SimpleTestToolService(_ToolService):
            def __call__(self, *args, **kwargs) -> Any:
                pass

        tool_service = SimpleTestToolService(
            name="test_tool", uri="https://api.test.com", api_key=Secret.from_token("initial_key")
        )

        self.assertEqual("initial_key", tool_service.api_key.resolve_value())

        # Test setter
        tool_service.api_key = Secret.from_token("new_key")
        self.assertEqual("new_key", tool_service.api_key.resolve_value())
        self.assertEqual("new_key", tool_service.secrets["api_key"].resolve_value())

        # Test setting to None
        tool_service.api_key = None
        self.assertIsNone(tool_service.api_key)

        with self.assertRaisesRegex(TypeError, "api_key"):
            tool_service.api_key = "not-logged"

        with self.assertRaisesRegex(TypeError, "api_key"):
            SimpleTestToolService(name="test", uri="https://api.test.com", api_key="not-logged")

    def test_api_key_property_with_secret(self):
        """The api_key getter returns a Secret unchanged; resolution happens at the consumer."""

        @register_tool_service("simple_format")
        class SimpleTestToolService(_ToolService):
            def __call__(self, *args, **kwargs) -> Any:
                pass

        secret = Secret.from_env_var("TOOL_API_KEY")
        tool_service = SimpleTestToolService(name="test_tool", uri="https://api.test.com", api_key=secret)
        self.assertIs(secret, tool_service.api_key)
        self.assertEqual({"api_key": secret}, tool_service.secrets)
        with patch.dict(os.environ, {"TOOL_API_KEY": "resolved-key"}):
            self.assertEqual("resolved-key", resolve_secret(tool_service.api_key))

    def test_as_tool_conversion(self):
        """Test conversion to a Tool instance"""

        @register_tool_service("simple_format")
        class SimpleTestToolService(_ToolService):
            """A simple tool service for testing."""

            def __call__(self, query: str, num_results: Optional[int] = 5) -> list[str]:
                """
                A simple search function.

                :param query: The search query
                :param num_results: Number of results to return
                :return: List of results
                """
                # Handle the None case explicitly to fix type checking
                count = 5 if num_results is None else num_results
                return [f"Result {i} for {query}" for i in range(count)]

        tool_service = SimpleTestToolService(name="test_tool", uri="https://api.test.com")

        # Test default conversion
        tool = tool_service.as_tool()
        self.assertIsInstance(tool, Tool)
        self.assertEqual("simple_test", tool.name)
        self.assertTrue(tool.description.startswith("A simple search function"))
        self.assertEqual({"query", "num_results"}, set(tool.parameters["properties"].keys()))
        self.assertEqual(5, tool.parameters["properties"]["num_results"]["default"])

        # Test with custom name and description
        custom_tool = tool_service.as_tool(name="custom_name", description="Custom description")
        self.assertEqual("custom_name", custom_tool.name)
        self.assertEqual("Custom description", custom_tool.description)

        # Test tool functionality
        results = custom_tool.function("test", num_results=3)
        self.assertEqual(["Result 0 for test", "Result 1 for test", "Result 2 for test"], results)

    def test_as_tool_with_varargs_raises_error(self):
        """Test that as_tool raises error with varargs in __call__"""

        @register_tool_service("varargs_format")
        class VarArgsToolService(_ToolService):
            """A tool service with variable arguments for testing."""

            def __call__(self, *args, **kwargs) -> Any:
                """A function that accepts variable arguments."""
                return args, kwargs

        tool_service = VarArgsToolService(name="varargs_tool", uri="https://api.test.com")

        with self.assertRaises(ValueError) as context:
            tool_service.as_tool()

        self.assertTrue(
            str(context.exception).startswith(
                "The __call__ method of VarArgsToolService must not accept *args or **kwargs"
            )
        )

    def test_tool_service_call(self):
        """Test tool service __call__ functionality"""

        @register_tool_service("simple_format")
        class SimpleTestToolService(_ToolService):
            def __call__(self, query: str, num_results: Optional[int] = 5) -> list[str]:
                count = 5 if num_results is None else num_results
                return [f"Result {i} for {query}" for i in range(count)]

        tool_service = SimpleTestToolService(name="test_tool", uri="https://api.test.com")

        results = tool_service("test query", num_results=2)
        self.assertEqual(["Result 0 for test query", "Result 1 for test query"], results)


if __name__ == "__main__":
    unittest.main()
