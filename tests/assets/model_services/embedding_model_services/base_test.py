import unittest
from unittest.mock import patch

import httpx

from ravelin.assets.model_services import register_model_service
from ravelin.assets.model_services.embedding_model_services import HttpxEmbeddingModelService


class TestCustomizedEmbeddingModelService(unittest.TestCase):
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
        register_model_service("custom_embedding_format")(self.CustomizedHttpxEmbeddingModelService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    @register_model_service("custom_embedding_format")
    class CustomizedHttpxEmbeddingModelService(HttpxEmbeddingModelService):
        def build_request(self, client, inputs, model=None, inference_kwargs=None, **kwargs):
            inference_kwargs = inference_kwargs or {}
            return client.build_request(
                "POST", url=self.base_url, json={"texts": inputs, "model": model, **inference_kwargs}, **kwargs
            )

        def parse_response(self, response):
            return response.json().get("embeddings", [])

    def test_initialization(self):
        customized_model_service = self.CustomizedHttpxEmbeddingModelService(
            name="test_customized_model_service", base_url="test_url"
        )
        self.assertEqual("test_customized_model_service", customized_model_service.name)
        self.assertEqual(
            {
                "base_url": "test_url",
                "organization": None,
                "inference_api_format": "custom_embedding_format",
            },
            customized_model_service.configs,
        )

    def test_client_kwargs(self):
        customized_model_service = self.CustomizedHttpxEmbeddingModelService(
            name="test_customized_model_service", base_url="test_url", client_kwargs={"headers": {"X-Test": "1"}}
        )
        self.assertEqual({"headers": {"X-Test": "1"}}, customized_model_service.configs["client_kwargs"])
        with patch("httpx.Client") as mock_client:
            _ = customized_model_service._client
            mock_client.assert_called_once_with(headers={"X-Test": "1"})
        self.assertEqual({}, customized_model_service.secrets)

    def test_infer(self):
        inputs = ["Hello"]
        customized_model_service = self.CustomizedHttpxEmbeddingModelService(
            name="test_customized_model_service", base_url="test_url"
        )
        # Patch the httpx client to return a response with embeddings
        with patch("httpx.Client") as mock_client:
            # Patch the response from the client
            mock_client.return_value.send.return_value.raise_for_status.return_value.json.return_value = {
                "embeddings": [[0.1, -0.2, 0.3]]
            }
            embeddings = customized_model_service.infer(
                inputs, model="custom-model", timeout=5.0, inference_kwargs={"param": "value"}
            )
            # Check that the client was called with the expected arguments
            mock_client.assert_called_once_with()
            # Check that the timeout was applied to the request
            mock_client.return_value.build_request.return_value.extensions.__setitem__.assert_called_once_with(
                "timeout", httpx.Timeout(5.0).as_dict()
            )
            # Check that the request was built with the expected arguments
            expected_request_json_body = {"texts": inputs, "model": "custom-model", "param": "value"}
            mock_client.return_value.build_request.assert_called_once_with(
                "POST",
                url="test_url",
                json=expected_request_json_body,
            )
            # Check that the request was sent
            mock_client.return_value.send.assert_called_once()
            mock_client.return_value.send.return_value.raise_for_status.assert_called_once()
            # Check that the response was parsed correctly
            self.assertEqual([[0.1, -0.2, 0.3]], embeddings)


if __name__ == "__main__":
    unittest.main()
