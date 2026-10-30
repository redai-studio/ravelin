import unittest
from typing import Any, List, Optional, Dict, Union
from unittest.mock import patch, Mock

from haystack import Document
from haystack.utils import Secret

from ravelin.assets.vector_databases._base import (
    _VectorDatabase,
    register_vector_database,
    VECTOR_DATABASE_REGISTRY,
)
from ravelin.dataclasses import EmbeddingField


# Define mock classes without decorators to avoid registering them globally
class MockMilvusDatabase(_VectorDatabase):
    def search_by_data(
        self,
        query_data: Dict[str, Any],
        *,
        embedding_fields: List[str],
        meta_fields: Optional[List[str]] = None,
        top_k: int,
        top_k_by_embedding_field: Optional[Union[int, Dict[str, int]]] = None,
        timeout: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        reranker: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        return [Document(content="milvus")]


class MockPineconeDatabase(_VectorDatabase):
    def search_by_data(
        self,
        query_data: Dict[str, Any],
        *,
        embedding_fields: List[str],
        meta_fields: Optional[List[str]] = None,
        top_k: int,
        top_k_by_embedding_field: Optional[Union[int, Dict[str, int]]] = None,
        timeout: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        reranker: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        return [Document(content="pinecone")]


class TestVectorDatabase(unittest.TestCase):
    """Tests for vector database functionality"""

    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.vector_databases._base.VECTOR_DATABASE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Now register our mock classes using the decorator function
        register_vector_database("milvus")(MockMilvusDatabase)
        register_vector_database("pinecone")(MockPineconeDatabase)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_registry_access(self):
        """Test basic registry functionality"""

        # Verify we can access the pre-registered mock classes
        registry = VECTOR_DATABASE_REGISTRY.get_all()
        self.assertIsInstance(registry, dict)
        self.assertEqual(2, len(registry))
        self.assertTrue(VECTOR_DATABASE_REGISTRY.contains("milvus"))
        self.assertTrue(VECTOR_DATABASE_REGISTRY.contains("pinecone"))

    def test_initialization(self):
        """Test vector database initialization with all parameters"""

        @register_vector_database("test_type")
        class CustomizedVectorDatabase(_VectorDatabase):
            def search_by_data(
                self,
                query_data: Dict[str, Any],
                *,
                embedding_fields: List[str],
                meta_fields: Optional[List[str]] = None,
                top_k: int,
                top_k_by_embedding_field: Optional[Union[int, Dict[str, int]]] = None,
                timeout: Optional[float] = None,
                filters: Optional[Dict[str, Any]] = None,
                reranker: Optional[Dict[str, Any]] = None,
                **kwargs,
            ) -> List[Document]:
                return [Document(content="abc")]

        vector_db = CustomizedVectorDatabase(
            name="test_db",
            collection_name="test_collection",
            embedding_fields={
                "field1": {
                    "from_asset": {"name": "abc", "source": "local"},
                    "embedder_kwargs": {"foo": "bar"},
                    "search_kwargs": {"foo": "baz"},
                }
            },
            content_field="content",
            meta_fields=["abc", "def"],
            uri="https://abc.com",
            host="localhost",
            port=19530,
            group_by={"field": "value"},
        )
        self.assertEqual("test_db", vector_db.name)
        self.assertDictEqual(
            {
                "vector_db_type": "test_type",
                "collection_name": "test_collection",
                "embedding_fields": {
                    "field1": {
                        "from_asset": {"name": "abc", "source": "local"},
                        "embedder_kwargs": {"foo": "bar"},
                        "search_kwargs": {"foo": "baz"},
                    }
                },
                "content_field": "content",
                "meta_fields": ["abc", "def"],
                "uri": "https://abc.com",
                "host": "localhost",
                "port": 19530,
                "group_by": {"field": "value"},
                "vector_db_name": None,
                "vector_db_user": None,
            },
            vector_db.configs,
        )
        self.assertDictEqual({"vector_db_password": None}, vector_db.secrets)

        self.assertEqual("content", vector_db.content_field)
        self.assertSetEqual({"abc", "def"}, vector_db.meta_fields)

        # Check the embedding fields were properly initialized
        self.assertIn("field1", vector_db.embedding_fields)
        embedding_field = vector_db.embedding_fields["field1"]
        self.assertIsInstance(embedding_field, EmbeddingField)
        # Check that search_kwargs were set correctly
        self.assertEqual({"foo": "baz"}, embedding_field.search_kwargs)
        # Since from_asset and embedder_kwargs are converted to embedding_function,
        # we need to check that _embedding_function_data is not None
        self.assertIsNotNone(embedding_field._embedding_function_data)

    def test_invalid_initialization_no_connection(self):
        """Test vector database initialization fails without connection details"""

        @register_vector_database("test_type")
        class CustomizedVectorDatabase(_VectorDatabase):
            def search_by_data(self, *args, **kwargs) -> List[Document]:
                return [Document(content="abc")]

        with self.assertRaisesRegex(
            ValueError, "Either a uri or host and port must be provided to connect to the vector database."
        ):
            CustomizedVectorDatabase(
                name="test_db",
                collection_name="test_collection",
                embedding_fields={},
                content_field="content",
            )

    def test_vector_db_password_requires_secret(self):
        """Test password construction and assignment enforce the Secret-only contract."""
        initial_secret = Secret.from_token("initial-password")
        vector_db = MockMilvusDatabase(
            name="test_db",
            collection_name="test_collection",
            embedding_fields={},
            content_field="content",
            uri="https://abc.com",
            vector_db_password=initial_secret,
        )
        self.assertIs(initial_secret, vector_db.vector_db_password)

        updated_secret = Secret.from_env_var("VECTOR_DB_PASSWORD")
        vector_db.vector_db_password = updated_secret
        self.assertIs(updated_secret, vector_db.vector_db_password)

        with self.assertRaisesRegex(TypeError, "vector_db_password"):
            vector_db.vector_db_password = "plaintext-password"
        self.assertIs(updated_secret, vector_db.vector_db_password)

        with self.assertRaisesRegex(TypeError, "vector_db_password"):
            MockMilvusDatabase(
                name="test_db",
                collection_name="test_collection",
                embedding_fields={},
                content_field="content",
                uri="https://abc.com",
                vector_db_password="plaintext-password",
            )

    def test_validate_meta_fields(self):
        """Test validation of meta fields against available fields"""

        @register_vector_database("test_type")
        class CustomizedVectorDatabase(_VectorDatabase):
            def search_by_data(self, *args, **kwargs) -> List[Document]:
                return [Document(content="abc")]

        vector_db = CustomizedVectorDatabase(
            name="test_db",
            collection_name="test_collection",
            embedding_fields={},
            content_field="content",
            meta_fields=["abc", "def"],
            uri="https://abc.com",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"meta_fields must be a subset of the available meta fields \(\{.+\}\) in the database, got: \['invalid'\]",
        ):
            vector_db.validate_meta_fields(["invalid"])

    def test_embed_query(self):
        """Test the _embed_query method."""

        @register_vector_database("test_type")
        class CustomizedVectorDatabase(_VectorDatabase):
            def search_by_data(self, *args, **kwargs) -> List[Document]:
                return [Document(content="abc")]

        # Create a vector database with embedding fields that have embedding functions
        vector_db = CustomizedVectorDatabase(
            name="test_db",
            collection_name="test_collection",
            embedding_fields={
                "dense_field": {"vector_type": "dense"},
                "sparse_field": {"vector_type": "sparse"},
            },
            content_field="content",
            uri="https://abc.com",
        )

        # Mock the embedding functions
        for field_name in vector_db.embedding_fields:
            embedding_field = vector_db.embedding_fields[field_name]
            if embedding_field.vector_type == "dense":
                embedding_field.embedding_function = Mock(return_value=[[0.1, -0.2]])

        # Test embedding a query
        query_data = vector_db._embed_query("test query", ["dense_field", "sparse_field"])
        self.assertEqual({"dense_field": [0.1, -0.2], "sparse_field": "test query"}, query_data)

        # Test embedding a query with a field that doesn't exist
        with self.assertRaisesRegex(ValueError, "Embedding field nonexistent not found in embedding_fields"):
            vector_db._embed_query("test query", ["nonexistent"])

    def test_search(self):
        """Test the search method which calls _embed_query and search_by_data."""

        @register_vector_database("test_type")
        class CustomizedVectorDatabase(_VectorDatabase):
            def search_by_data(self, *args, **kwargs) -> List[Document]:
                return [Document(content="search result")]

        vector_db = CustomizedVectorDatabase(
            name="test_db",
            collection_name="test_collection",
            embedding_fields={
                "field1": {"vector_type": "dense"},
            },
            content_field="content",
            uri="https://abc.com",
        )

        # Mock the _embed_query method
        vector_db._embed_query = Mock(return_value={"field1": [0.1, 0.2, 0.3]})

        # Call the search method
        results = vector_db.search(
            "test query",
            embedding_fields=["field1"],
            top_k=5,
        )

        # Assert that _embed_query was called with the correct arguments
        vector_db._embed_query.assert_called_once_with("test query", ["field1"])

        # Assert the search results
        self.assertEqual(1, len(results))
        self.assertEqual("search result", results[0].content)


if __name__ == "__main__":
    unittest.main()
