import unittest
from typing import Any
from unittest.mock import patch

from haystack.utils import Secret

from ravelin.assets.model_services._base import (
    _HttpModelService,
    _ModelService,
    register_model_service,
    MODEL_SERVICE_REGISTRY,
)


# Define mock classes without decorators to avoid registering them globally
class MockOpenAIService(_ModelService):
    def infer(self, *args, **kwargs) -> Any:
        pass


class MockAnthropicService(_ModelService):
    def infer(self, *args, **kwargs) -> Any:
        pass


class TestModelService(unittest.TestCase):
    """Tests for model service functionality"""

    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Now register our mock classes using the decorator function
        register_model_service("openai")(MockOpenAIService)
        register_model_service("anthropic")(MockAnthropicService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_initialization_with_decorator(self):
        """Test initialization with decorator-registered class"""

        # Define the class after the registry is patched
        @register_model_service("decorator_format")
        class DecoratorModelService(_ModelService):
            def infer(self, *args, **kwargs) -> Any:
                pass

        # Verify class is in registry
        self.assertTrue(MODEL_SERVICE_REGISTRY.contains("decorator_format"))
        self.assertEqual(DecoratorModelService, MODEL_SERVICE_REGISTRY.get("decorator_format"))

        model_service = DecoratorModelService(name="test_service")
        self.assertEqual("test_service", model_service.name)
        self.assertEqual(
            {
                "organization": None,
                "inference_api_format": "decorator_format",
            },
            model_service.configs,
        )

    def test_initialization_all_args(self):
        """Test initialization with all arguments"""

        @register_model_service("decorator_format")
        class DecoratorModelService(_ModelService):
            def infer(self, *args, **kwargs) -> Any:
                pass

        model_service = DecoratorModelService(
            name="test_service",
            api_key=Secret.from_token("test_api_key"),
            organization="test_org",
        )
        self.assertEqual("test_service", model_service.name)
        self.assertEqual(
            {
                "organization": "test_org",
                "inference_api_format": "decorator_format",
            },
            model_service.configs,
        )
        self.assertEqual("test_api_key", model_service.secrets["api_key"].resolve_value())

    def test_api_key_getter_setter(self):
        """Test api_key property getter and setter"""

        @register_model_service("test_format")
        class TestModelService(_ModelService):
            def infer(self, *args, **kwargs) -> Any:
                pass

        # Test with api_key at initialization
        model_service = TestModelService(name="test", api_key=Secret.from_token("initial_key"))
        self.assertEqual("initial_key", model_service.api_key.resolve_value())
        self.assertEqual("initial_key", model_service.secrets["api_key"].resolve_value())

        # Test setting api_key
        model_service.api_key = Secret.from_token("updated_key")
        self.assertEqual("updated_key", model_service.api_key.resolve_value())
        self.assertEqual("updated_key", model_service.secrets["api_key"].resolve_value())

        # Test setting to None
        model_service.api_key = None
        self.assertIsNone(model_service.api_key)
        self.assertEqual({"api_key": None}, model_service.secrets)

        with self.assertRaisesRegex(TypeError, "api_key"):
            model_service.api_key = "not-logged"

        with self.assertRaisesRegex(TypeError, "api_key"):
            TestModelService(name="test", api_key="not-logged")

    def test_to_dict(self):
        """Test conversion to dictionary"""

        @register_model_service("decorator_format")
        class DecoratorModelService(_ModelService):
            def infer(self, *args, **kwargs) -> Any:
                pass

        model_service = DecoratorModelService(name="test_service")
        model_service_dict = model_service.to_dict()

        expected_dict = {
            "name": "test_service",
            "type": "model-service",
            "configs": {
                "organization": None,
                "inference_api_format": "decorator_format",
            },
            "secrets": {},
        }

        self.assertEqual(expected_dict, model_service_dict)

    def test_registry_access(self):
        """Test basic registry functionality"""

        # Verify we can access the pre-registered mock classes
        registry = MODEL_SERVICE_REGISTRY.get_all()
        self.assertIsInstance(registry, dict)
        self.assertEqual(2, len(registry))
        self.assertTrue(MODEL_SERVICE_REGISTRY.contains("openai"))
        self.assertTrue(MODEL_SERVICE_REGISTRY.contains("anthropic"))

    def test_generic_model_service_does_not_require_an_endpoint(self):
        """A non-HTTP model service has no endpoint requirement."""

        @register_model_service("test_format")
        class TestModelService(_ModelService):
            def infer(self, *args, **kwargs) -> Any:
                pass

        model_service = TestModelService(name="test_service")

        self.assertNotIn("base_url", model_service.configs)

    def test_http_model_service_routes(self):
        """HTTP services merge per-instance route overrides over class defaults."""

        @register_model_service("http_test_format")
        class TestHttpModelService(_HttpModelService):
            DEFAULT_ROUTES = {"inference": "/v1/predict", "stream": "/v1/stream"}

            def infer(self, *args, **kwargs) -> Any:
                pass

        model_service = TestHttpModelService(
            name="test_service",
            base_url="https://example.com/deployment/",
            routes={"inference": "custom/predict"},
        )

        self.assertEqual("https://example.com/deployment", model_service.base_url)
        self.assertEqual("custom/predict", model_service.routes["inference"])
        self.assertEqual("/v1/stream", model_service.routes["stream"])
        self.assertEqual(
            "https://example.com/deployment/custom/predict",
            model_service.build_request_url("inference"),
        )
        self.assertEqual(
            {"base_url": "https://example.com/deployment/", "routes": {"inference": "custom/predict"}},
            {key: model_service.configs[key] for key in ("base_url", "routes")},
        )

        with self.assertRaises(TypeError):
            model_service.routes["inference"] = "/other"  # type: ignore[index]

    def test_http_model_service_requires_base_url(self):
        """Only HTTP-backed services require a base URL."""

        @register_model_service("http_test_format")
        class TestHttpModelService(_HttpModelService):
            def infer(self, *args, **kwargs) -> Any:
                pass

        with self.assertRaisesRegex(ValueError, "base URL"):
            TestHttpModelService(name="test_service")

        with self.assertRaisesRegex(ValueError, "base URL"):
            TestHttpModelService(name="test_service", base_url="   ")


if __name__ == "__main__":
    unittest.main()
