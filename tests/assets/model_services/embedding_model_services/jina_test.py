import unittest
from unittest.mock import Mock

from haystack.utils import Secret
from httpx import Response

from ravelin.assets.model_services.embedding_model_services import JinaEmbeddingModelService


class TestJinaEmbeddingModelService(unittest.TestCase):
    def setUp(self):
        self.model_service = JinaEmbeddingModelService(
            name="test_jina_model_service", base_url="https://jina/url/", api_key=Secret.from_token("test_api_key")
        )

    def test_initialization(self):
        self.assertEqual(self.model_service.name, "test_jina_model_service")
        self.assertDictEqual(
            {
                "base_url": "https://jina/url/",
                "organization": None,
                "inference_api_format": "jina-embedding",
            },
            self.model_service.configs,
        )
        self.assertEqual("test_api_key", self.model_service.secrets["api_key"].resolve_value())

    def test_initialization_all_args(self):
        jina_model_service = JinaEmbeddingModelService(
            name="test_jina_model_service",
            base_url="test_uri",
            organization="test_org",
            api_key=Secret.from_token("test_api_key"),
        )
        self.assertEqual("test_jina_model_service", jina_model_service.name)
        self.assertDictEqual(
            {
                "base_url": "test_uri",
                "organization": "test_org",
                "inference_api_format": "jina-embedding",
            },
            jina_model_service.configs,
        )
        self.assertEqual("test_api_key", jina_model_service.secrets["api_key"].resolve_value())

    def test_build_request(self):
        client = Mock()
        messages = ["Hello"]
        self.model_service.build_request(
            client, messages, model="jina-embeddings-v2-base-zh", inference_kwargs={"encoding_format": "float"}
        )

        expected_json_body = {"model": "jina-embeddings-v2-base-zh", "input": ["Hello"], "encoding_format": "float"}

        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_api_key",
        }

        client.build_request.assert_called_once_with(
            "POST",
            url="https://jina/url/v1/embeddings",
            json=expected_json_body,
            headers=expected_headers,
        )

    def test_parse_response(self):
        response_json = {
            "data": [
                {"index": 0, "embedding": [0.1, -0.2, 0.3], "object": "embedding"},
                {"index": 1, "embedding": [-0.3, 0.2, 0.1], "object": "embedding"},
            ],
            "usage": {"total_tokens": 15, "prompt_tokens": 15},
        }
        embeddings = self.model_service.parse_response(Response(200, json=response_json))
        self.assertEqual(2, len(embeddings))
        self.assertListEqual([0.1, -0.2, 0.3], embeddings[0])
        self.assertListEqual([-0.3, 0.2, 0.1], embeddings[1])


if __name__ == "__main__":
    unittest.main()
