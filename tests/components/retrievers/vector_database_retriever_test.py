import unittest
from unittest.mock import Mock, patch, MagicMock

from haystack import Document

from ravelin.assets.vector_databases import VectorDatabase
from ravelin.components.retrievers.vector_database_retriever import VectorDatabaseRetriever
from ravelin.dataclasses import EmbeddingField
from ravelin.dataclasses.asset_reference import AssetReference


class TestVectorDatabaseRetriever(unittest.TestCase):
    def setUp(self):
        # Create the mock vector database
        self.vector_database = Mock(spec=VectorDatabase)
        self.vector_database.reference = AssetReference(name="test-database", source="local")
        self.vector_database.meta_fields = {"123", "456"}

        # Mock the embedding fields
        self.vector_database.embedding_fields = {
            "test_field1": Mock(spec=EmbeddingField),
            "test_field2": Mock(spec=EmbeddingField),
        }

        # Create mock embedder functions
        dense_embedder = MagicMock(return_value=[[0.1, -0.2, 0.3]])
        sparse_embedder = MagicMock(return_value=[[0.4, -0.5, 0.6]])

        # Mock the get_embedder method on the vector database
        def get_mock_embedder(field_name):
            if field_name == "test_field1":
                return dense_embedder
            elif field_name == "test_field2":
                return sparse_embedder
            return None

        self.vector_database.get_embedder = Mock(side_effect=get_mock_embedder)
        self.vector_database.search.return_value = [Document(content="abc", meta={"field": "value"})]

    def test_init(self):
        retriever = VectorDatabaseRetriever(
            vector_database=self.vector_database,
            meta_fields=["123"],
            top_k=10,
            top_k_by_embedding_field={"test_field1": 5, "test_field2": 10},
            timeout=20.0,
            filters={"expr": "value > 0"},
            reranker={"strategy": "rrf", "params": {"k": 80}},
            retrieval_kwargs={"key": "value"},
        )
        self.assertEqual(self.vector_database, retriever.vector_database)
        self.assertEqual(["123"], retriever.meta_fields)
        self.assertEqual(["test_field1", "test_field2"], retriever.embedding_fields)
        self.assertEqual(10, retriever.top_k)
        self.assertDictEqual({"test_field1": 5, "test_field2": 10}, retriever.top_k_by_embedding_field)
        self.assertEqual(20.0, retriever.timeout)
        self.assertDictEqual({"expr": "value > 0"}, retriever.filters)
        self.assertDictEqual({"strategy": "rrf", "params": {"k": 80}}, retriever.reranker)
        self.assertDictEqual({"key": "value"}, retriever.retrieval_kwargs)

        self.vector_database.validate_meta_fields.assert_called_once_with(["123"])

    def test_to_dict(self):
        retriever_dict = VectorDatabaseRetriever(
            vector_database=self.vector_database, meta_fields=["meta_field"], filters={"expr": "value > 0"}
        ).to_dict()

        self.assertIn("init_parameters", retriever_dict)
        self.assertIn("type", retriever_dict)
        self.assertEqual(
            "ravelin.components.retrievers.vector_database_retriever.VectorDatabaseRetriever",
            retriever_dict["type"],
        )
        self.assertDictEqual(
            self.vector_database.reference.to_dict(), retriever_dict["init_parameters"]["vector_database"]
        )
        self.assertListEqual(["meta_field"], retriever_dict["init_parameters"]["meta_fields"])
        self.assertListEqual(["test_field1", "test_field2"], retriever_dict["init_parameters"]["embedding_fields"])
        self.assertEqual(10, retriever_dict["init_parameters"]["top_k"])
        self.assertIsNone(retriever_dict["init_parameters"]["top_k_by_embedding_field"])
        self.assertIsNone(retriever_dict["init_parameters"]["timeout"])
        self.assertDictEqual({"expr": "value > 0"}, retriever_dict["init_parameters"]["filters"])
        self.assertIsNone(retriever_dict["init_parameters"]["reranker"])
        self.assertDictEqual({}, retriever_dict["init_parameters"]["retrieval_kwargs"])

    @patch("ravelin.components.retrievers.vector_database_retriever.VectorDatabaseFactory.load_from_reference")
    def test_from_dict(self, mock_load_vdb_from_reference):
        mock_load_vdb_from_reference.return_value = self.vector_database

        data = {
            "type": "ravelin.components.retrievers.vector_database_retriever.VectorDatabaseRetriever",
            "init_parameters": {
                "vector_database": {
                    "name": "test-database",
                    "source": "local",
                },
                "meta_fields": ["meta_field"],
                "embedding_fields": ["test_field1", "test_field2"],
                "top_k": 5,
                "top_k_by_embedding_field": {"test_field1": 5, "test_field2": 10},
                "timeout": 10.0,
                "filters": {"expr": "value > 0"},
                "reranker": {"strategy": "rrf", "params": {"k": 80}},
                "retrieval_kwargs": {"key": "value"},
            },
        }
        retriever = VectorDatabaseRetriever.from_dict(data)

        mock_load_vdb_from_reference.assert_called_once_with(
            AssetReference.from_dict(data["init_parameters"]["vector_database"])
        )
        self.assertIsInstance(retriever, VectorDatabaseRetriever)
        self.assertEqual(self.vector_database, retriever.vector_database)
        self.assertListEqual(["meta_field"], retriever.meta_fields)
        self.assertListEqual(["test_field1", "test_field2"], retriever.embedding_fields)
        self.assertEqual(5, retriever.top_k)
        self.assertDictEqual({"test_field1": 5, "test_field2": 10}, retriever.top_k_by_embedding_field)
        self.assertEqual(10.0, retriever.timeout)
        self.assertDictEqual({"expr": "value > 0"}, retriever.filters)
        self.assertDictEqual({"strategy": "rrf", "params": {"k": 80}}, retriever.reranker)
        self.assertDictEqual({"key": "value"}, retriever.retrieval_kwargs)

    def test_init_with_invalid_embedding_field(self):
        with self.assertRaises(ValueError):
            VectorDatabaseRetriever(vector_database=self.vector_database, embedding_fields=["invalid_field"])

    def test_run(self):
        retriever = VectorDatabaseRetriever(
            vector_database=self.vector_database,
            meta_fields=[],
            top_k=10,
            top_k_by_embedding_field=15,
            filters={"expr": "value > 0"},
            retrieval_kwargs={"search_foo": 123, "search_bar": 456},
        )

        # Run the retriever with a text query
        result = retriever.run(
            "test query",
            filters={"dataset_id": -1},  # Add a new filter
            retrieval_kwargs={"search_bar": 123},  # Overwrite the default value
        )

        # Check that the search method is called with the correct arguments
        # The query data should now include both fields
        self.vector_database.search.assert_called_once_with(
            query="test query",
            embedding_fields=["test_field1", "test_field2"],
            meta_fields=[],
            top_k=10,
            top_k_by_embedding_field=15,
            timeout=None,
            filters={"expr": "value > 0", "dataset_id": -1},
            reranker=None,
            search_foo=123,
            search_bar=123,
        )

        # Check that the result is returned correctly
        self.assertIn("documents", result)
        self.assertListEqual(
            [Document(content="abc", meta={"field": "value"})], self.vector_database.search.return_value
        )


if __name__ == "__main__":
    unittest.main()
