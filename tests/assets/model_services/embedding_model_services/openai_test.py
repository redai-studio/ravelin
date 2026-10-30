import unittest
from unittest.mock import Mock

from haystack.utils import Secret
from httpx import Response

from ravelin.assets.model_services.embedding_model_services import OpenAIEmbeddingModelService


class TestOpenAIEmbeddingModelService(unittest.TestCase):
    def setUp(self):
        self.model_service = OpenAIEmbeddingModelService(
            name="test_openai_model_service", base_url="https://openai/url/", api_key=Secret.from_token("test_api_key")
        )

    def test_initialization(self):
        self.assertEqual(self.model_service.name, "test_openai_model_service")
        self.assertDictEqual(
            {
                "base_url": "https://openai/url/",
                "organization": None,
                "inference_api_format": "openai-embedding",
            },
            self.model_service.configs,
        )
        self.assertEqual("test_api_key", self.model_service.secrets["api_key"].resolve_value())

    def test_initialization_all_args(self):
        openai_model_service = OpenAIEmbeddingModelService(
            name="test_openai_model_service",
            base_url="test_uri",
            organization="test_org",
            api_key=Secret.from_token("test_api_key"),
        )
        self.assertEqual("test_openai_model_service", openai_model_service.name)
        self.assertDictEqual(
            {
                "base_url": "test_uri",
                "organization": "test_org",
                "inference_api_format": "openai-embedding",
            },
            openai_model_service.configs,
        )
        self.assertEqual("test_api_key", openai_model_service.secrets["api_key"].resolve_value())

    def test_build_request(self):
        client = Mock()
        inputs = ["Hello"]
        self.model_service.build_request(
            client, inputs, model="text-embedding-3-large", inference_kwargs={"encoding_format": "float"}
        )

        expected_json_body = {"model": "text-embedding-3-large", "input": ["Hello"], "encoding_format": "float"}

        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_api_key",
            "api-key": "test_api_key",
        }

        client.build_request.assert_called_once_with(
            "POST",
            url="https://openai/url/v1/embeddings",
            json=expected_json_body,
            headers=expected_headers,
        )

    def test_build_request_with_invalid_input_type(self):
        client = Mock()
        inputs = ["Hello", 123]  # Mix of string and non-string

        with self.assertRaises(ValueError) as context:
            self.model_service.build_request(client, inputs)

        self.assertEqual(
            "Input 123 is not a string and cannot be embedded in OpenAI Embedding API format", str(context.exception)
        )

    def test_parse_response(self):
        response_json = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, -0.2, 0.3],
                    "index": 0,
                },
                {
                    "object": "embedding",
                    "embedding": [-0.3, 0.2, 0.1],
                    "index": 1,
                },
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }
        embeddings = self.model_service.parse_response(Response(200, json=response_json))
        self.assertEqual(2, len(embeddings))
        self.assertListEqual([0.1, -0.2, 0.3], embeddings[0])
        self.assertListEqual([-0.3, 0.2, 0.1], embeddings[1])


if __name__ == "__main__":
    unittest.main()
