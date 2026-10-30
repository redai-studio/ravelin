import os
import unittest
from unittest.mock import patch, Mock

from haystack import Document
from haystack.utils import Secret
from pymilvus import Hit, RRFRanker, WeightedRanker

from ravelin.assets.vector_databases import MilvusVectorDatabase
from ravelin.dataclasses import EmbeddingField
from ravelin.dataclasses.asset_reference import AssetReference


class TestMilvusVectorDatabase(unittest.TestCase):
    def setUp(self):
        with (
            patch("ravelin.assets.vector_databases.milvus.connections") as self.patch_connections,
            patch("ravelin.assets.vector_databases.milvus.Collection") as self.patch_collection,
        ):
            self.vector_db = MilvusVectorDatabase(
                name="test_db",
                collection_name="test_collection",
                uri="https://test_connection.milvus.com/",
                host="8.8.8.8",
                port=1234,
                embedding_fields={
                    "field1": {
                        "from_asset": {"name": "abc", "source": "local"},
                        "embedder_kwargs": {"model_name": "tao-8k"},
                        "search_kwargs": {"metric_type": "COSINE"},
                    }
                },
                content_field="text",
                meta_fields=["foo", "bar"],
                group_by={"field": "group"},
            )

    def test_initialization(self):
        self.patch_connections.connect.assert_called_once_with(
            alias="default",
            uri="https://test_connection.milvus.com/",
            host="8.8.8.8",
            port=1234,
            user="",
            password="",
            db_name="default",
        )
        self.patch_collection.assert_called_once_with("test_collection")
        self.assertEqual(self.patch_collection.return_value, self.vector_db.collection)

        self.assertEqual("test_db", self.vector_db.name)
        self.assertDictEqual(
            {
                "vector_db_type": "milvus",
                "collection_name": "test_collection",
                "uri": "https://test_connection.milvus.com/",
                "host": "8.8.8.8",
                "port": 1234,
                "embedding_fields": {
                    "field1": {
                        "from_asset": {"name": "abc", "source": "local"},
                        "embedder_kwargs": {"model_name": "tao-8k"},
                        "search_kwargs": {"metric_type": "COSINE"},
                    }
                },
                "content_field": "text",
                "meta_fields": ["foo", "bar"],
                "group_by": {"field": "group"},
                "vector_db_name": None,
                "vector_db_user": None,
            },
            self.vector_db.configs,
        )
        self.assertDictEqual(
            {
                "field1": EmbeddingField(
                    from_asset=AssetReference(name="abc", source="local"),
                    embedder_kwargs={"model_name": "tao-8k"},
                    search_kwargs={"metric_type": "COSINE"},
                )
            },
            self.vector_db.embedding_fields,
        )
        self.assertEqual("text", self.vector_db.content_field)
        self.assertSetEqual({"foo", "bar"}, self.vector_db.meta_fields)

    def test_secret_password_resolved_at_connect(self):
        secret = Secret.from_env_var("MILVUS_PASSWORD")
        with (
            patch("ravelin.assets.vector_databases.milvus.connections") as patch_connections,
            patch("ravelin.assets.vector_databases.milvus.Collection"),
            patch.dict(os.environ, {"MILVUS_PASSWORD": "resolved-password"}),
        ):
            vector_db = MilvusVectorDatabase(
                name="test_db",
                collection_name="test_collection",
                uri="https://test_connection.milvus.com/",
                embedding_fields={
                    "field1": {
                        "from_asset": {"name": "abc", "source": "local"},
                        "embedder_kwargs": {"model_name": "tao-8k"},
                        "search_kwargs": {"metric_type": "COSINE"},
                    }
                },
                content_field="text",
                vector_db_password=secret,
            )
        # The Secret is resolved to a plain string before it reaches pymilvus.
        self.assertEqual("resolved-password", patch_connections.connect.call_args.kwargs["password"])
        # The Secret itself is retained on the asset (serialized safely by to_dict).
        self.assertIs(secret, vector_db.vector_db_password)

    def test_internal_search_single_field(self):
        query_values = [[1.0, -2.0, 3.0]]
        results = self.vector_db._search(
            query_values,
            anns_field="field1",
            output_fields=["text", "bar"],
            top_k=10,
            timeout=10.0,
            filters={"expr": "value > 0"},
            partition_names=["partition1", "partition2"],
            round_decimal=4,
        )
        self.patch_collection.return_value.search.assert_called_once_with(
            data=[[1.0, -2.0, 3.0]],
            anns_field="field1",
            param={"metric_type": "COSINE"},
            limit=10,
            expr="value > 0",
            partition_names=["partition1", "partition2"],
            output_fields=["text", "bar"],
            timeout=10.0,
            round_decimal=4,
            group_by_field="group",
        )
        self.assertEqual(self.patch_collection.return_value.search.return_value, results)

    def test_internal_search_single_field_invalid_anns_field(self):
        query_values = [[1.0, 2.0, 3.0]]
        with self.assertRaisesRegex(ValueError, "not found"):
            self.vector_db._search(query_values, anns_field="other_field", top_k=10, output_fields=["text"])

    @patch("ravelin.assets.vector_databases.milvus.AnnSearchRequest")
    def test_build_search_request_by_field(self, patch_ann_search_request):
        query_values = [[1.0, -2.0, 3.0]]
        self.vector_db._build_search_request_by_field(
            query_values,
            anns_field="field1",
            top_k=10,
            filters={"expr": "value > 0"},
        )
        patch_ann_search_request.assert_called_once_with(
            data=[[1.0, -2.0, 3.0]],
            anns_field="field1",
            param={"metric_type": "COSINE"},
            limit=10,
            expr="value > 0",
        )

    def test_build_search_request_by_field_invalid_anns_field(self):
        query_values = [[1.0, 2.0, 3.0]]
        with self.assertRaisesRegex(ValueError, "not found"):
            self.vector_db._build_search_request_by_field(query_values, anns_field="other_field", top_k=10)

    def test_load_reranker(self):
        default_reranker = self.vector_db._load_reranker(None)
        self.assertEqual({"strategy": "rrf", "params": {"k": 60}}, default_reranker.dict())

        test_cases = [
            ({"strategy": "rrf", "params": {"k": 80}}, RRFRanker, None),
            ({"strategy": "weighted", "params": {"weights": [0.5, 0.5], "norm_score": True}}, WeightedRanker, None),
            ({"strategy": "invalid", "params": {}}, None, (ValueError, "Invalid reranker strategy: invalid")),
        ]
        for reranker_config, reranker_class, exception in test_cases:
            with self.subTest(reranker_config=reranker_config, reranker_class=reranker_class, exception=exception):
                if exception is not None:
                    with self.assertRaisesRegex(exception[0], exception[1]):
                        self.vector_db._load_reranker(reranker_config)
                else:
                    reranker = self.vector_db._load_reranker(reranker_config)
                    self.assertIsInstance(reranker, reranker_class)
                    self.assertEqual(reranker_config, reranker.dict())

    @patch("ravelin.assets.vector_databases.milvus.MilvusVectorDatabase._search")
    def test_search_single_field(self, patch_search):
        patch_search.return_value = [[Hit({"distance": 0.6, "entity": {"text": "result"}}, pk_name="123")]]
        query_data = {"field1": [0.1, -0.2, 0.3]}
        results = self.vector_db.search_by_data(
            query_data=query_data,
            embedding_fields=["field1"],
            meta_fields=[],
            top_k=10,
            timeout=10.0,
            filters={"expr": "value > 0"},
            partition_names=["partition1", "partition2"],
            round_decimal=4,
        )
        patch_search.assert_called_once_with(
            query_values=[query_data["field1"]],
            anns_field="field1",
            output_fields=["text"],
            top_k=10,
            timeout=10.0,
            filters={"expr": "value > 0"},
            partition_names=["partition1", "partition2"],
            round_decimal=4,
        )
        self.assertIsInstance(results[0], Document)
        self.assertEqual("result", results[0].content)
        self.assertDictEqual({"distance": 0.6}, results[0].meta)

    @patch("ravelin.assets.vector_databases.milvus.connections")
    @patch("ravelin.assets.vector_databases.milvus.MilvusVectorDatabase._load_reranker")
    @patch("ravelin.assets.vector_databases.milvus.Collection")
    @patch("ravelin.assets.vector_databases.milvus.MilvusVectorDatabase._build_search_request_by_field")
    def test_search_multiple_fields(self, patch_search_request_by_field, patch_collection, patch_load_reranker, _):
        patch_collection.return_value.hybrid_search.return_value = [
            [
                Hit({"distance": 0.6, "entity": {"output": "123", "foo": "baz", "group": "1"}}, pk_name="123"),
                Hit({"distance": 0.4, "entity": {"output": "321", "bar": "baz", "group": "2"}}, pk_name="321"),
            ]
        ]
        patch_search_request_by_field.return_value = Mock()
        vector_db = MilvusVectorDatabase(
            name="test_db",
            collection_name="test_collection",
            host="localhost",
            port=19530,
            embedding_fields={
                "field1": {"vector_type": "dense", "from_asset": {"name": "abc", "source": "local"}},
                "field2": {"vector_type": "sparse"},
            },
            content_field="output",
            meta_fields=["foo", "bar"],
            group_by={"field": "group"},
        )
        query_data = {"field1": [0.1, -0.2, 0.3], "field2": "search query text"}
        anns_fields = ["field1", "field2"]

        results = vector_db.search_by_data(
            query_data=query_data,
            embedding_fields=anns_fields,
            meta_fields=["foo", "bar"],
            top_k=10,
            top_k_by_embedding_field={"field1": 50},
            timeout=10.0,
            filters={"expr": "value > 0"},
            reranker={"strategy": "rrf", "params": {"k": 60}},
            partition_names=["partition1", "partition2"],
            round_decimal=4,
        )
        patch_load_reranker.assert_called_once_with({"strategy": "rrf", "params": {"k": 60}})
        for anns_field in anns_fields:
            patch_search_request_by_field.assert_any_call(
                query_values=[query_data[anns_field]],
                anns_field=anns_field,
                top_k=(50 if anns_field == "field1" else 10),
                filters={"expr": "value > 0"},
            )
        patch_collection.return_value.hybrid_search.assert_called_once_with(
            reqs=[patch_search_request_by_field.return_value, patch_search_request_by_field.return_value],
            rerank=patch_load_reranker.return_value,
            limit=10,
            partition_names=["partition1", "partition2"],
            output_fields=["output", "foo", "bar"],
            timeout=10.0,
            round_decimal=4,
            group_by_field="group",
        )

        self.assertIsInstance(results[0], Document)
        self.assertEqual("123", results[0].content)
        self.assertDictEqual({"foo": "baz", "distance": 0.6, "group": "1"}, results[0].meta)
        self.assertIsInstance(results[1], Document)
        self.assertEqual("321", results[1].content)
        self.assertDictEqual({"bar": "baz", "distance": 0.4, "group": "2"}, results[1].meta)
        self.assertEqual(2, len(results))

    def test_search_no_embedding_fields(self):
        with self.assertRaisesRegex(ValueError, "embedding_fields must not be empty"):
            self.vector_db.search_by_data(
                query_data={},
                embedding_fields=[],
                top_k=10,
            )

    @patch("ravelin.assets.vector_databases.milvus.connections")
    @patch("ravelin.assets.vector_databases.milvus.Collection")
    def test_connection_failure_with_uri(self, patch_collection, patch_connections):
        """Test that connection failures with URI raise a clear ConnectionError."""
        patch_connections.connect.side_effect = Exception("UNAUTHENTICATED: missing authorization in header")

        with self.assertRaisesRegex(
            ConnectionError,
            r"Failed to connect to Milvus database \(uri='http://localhost:19530'\)\. "
            r"Error: UNAUTHENTICATED: missing authorization in header\. "
            r"Please verify your connection parameters and credentials\.",
        ):
            MilvusVectorDatabase(
                collection_name="test_collection",
                uri="http://localhost:19530",
                embedding_fields={"field1": {}},
                content_field="text",
            )

    @patch("ravelin.assets.vector_databases.milvus.connections")
    @patch("ravelin.assets.vector_databases.milvus.Collection")
    def test_connection_failure_with_host_port(self, patch_collection, patch_connections):
        """Test that connection failures with host/port raise a clear ConnectionError."""
        patch_connections.connect.side_effect = Exception("Connection timeout")

        with self.assertRaisesRegex(
            ConnectionError,
            r"Failed to connect to Milvus database \(host='localhost', port=19530\)\. "
            r"Error: Connection timeout\. "
            r"Please verify your connection parameters and credentials\.",
        ):
            MilvusVectorDatabase(
                collection_name="test_collection",
                host="localhost",
                port=19530,
                embedding_fields={"field1": {}},
                content_field="text",
            )

    @patch("ravelin.assets.vector_databases.milvus.connections")
    @patch("ravelin.assets.vector_databases.milvus.Collection")
    def test_initialization_with_auth_params(self, patch_collection, patch_connections):
        """Test initialization with authentication parameters."""
        vector_db = MilvusVectorDatabase(
            name="test_db",
            collection_name="test_collection",
            uri="http://localhost:19530",
            vector_db_name="custom_db",
            vector_db_user="admin",
            vector_db_password=Secret.from_token("secret123"),
            embedding_fields={"field1": {}},
            content_field="text",
        )

        # Verify connection was called with auth parameters
        patch_connections.connect.assert_called_once_with(
            alias="default",
            uri="http://localhost:19530",
            host="",
            port="",
            user="admin",
            password="secret123",
            db_name="custom_db",
        )

        # Verify password is stored in secrets, not configs
        self.assertIn("vector_db_password", vector_db.secrets)
        self.assertEqual("secret123", vector_db.secrets["vector_db_password"].resolve_value())
        self.assertNotIn("vector_db_password", vector_db.configs)

        # Verify username and db_name are in configs
        self.assertEqual("admin", vector_db.configs["vector_db_user"])
        self.assertEqual("custom_db", vector_db.configs["vector_db_name"])


if __name__ == "__main__":
    unittest.main()
