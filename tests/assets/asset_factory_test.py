import unittest
from typing import Any
from unittest.mock import patch

from haystack.utils import Secret

from ravelin.assets.asset_factory import (
    ModelServiceFactory,
    ToolServiceFactory,
    VectorDatabaseFactory,
    AssetFactory,
)
from ravelin.assets.model_services import ModelService, register_model_service
from ravelin.assets.tool_services import ToolService, register_tool_service
from ravelin.assets.vector_databases import VectorDatabase, register_vector_database
from ravelin.dataclasses.asset_reference import AssetReference


# Define mock classes without decorators to avoid registering them globally
class TestModelService(ModelService):
    __test__ = False

    def infer(self, *args, **kwargs) -> Any:
        pass


class TestToolService(ToolService):
    __test__ = False

    def __call__(self, *args, **kwargs) -> Any:
        pass


class TestVectorDatabase(VectorDatabase):
    __test__ = False

    def search_by_data(
        self,
        query_data,
        *,
        embedding_fields,
        top_k,
        meta_fields=None,
        top_k_by_embedding_field=None,
        timeout=None,
        filters=None,
        reranker=None,
        **kwargs,
    ):
        pass


class TestModelServiceFactory(unittest.TestCase):
    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Now register our mock class using the decorator function
        register_model_service("test_format")(TestModelService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_create_model_service_with_supported_format(self):
        """Test creating model service with supported format"""
        model_service = ModelServiceFactory.create("test_format", name="test_model_service")
        self.assertIsInstance(model_service, TestModelService)

    def test_create_model_service_with_unsupported_format(self):
        """Test creating model service with unsupported format raises error"""
        with self.assertRaisesRegex(ValueError, "Unsupported inference_api_format: unsupported_format"):
            ModelServiceFactory.create("unsupported_format", name="test_model_service")

    def test_load_model_service_from_dict_with_supported_format(self):
        """Test loading model service from dictionary with supported format"""
        model_service = ModelServiceFactory.load_from_dict(
            {
                "name": "test_model_service",
                "type": "ModelService",
                "configs": {"inference_api_format": "test_format"},
                "secrets": {"api_key": Secret.from_token("test_key")},
            }
        )
        self.assertIsInstance(model_service, TestModelService)
        self.assertNotIn("base_url", model_service.configs)

    def test_load_model_service_from_dict_with_unsupported_format(self):
        """Test loading model service from dictionary with unsupported format raises error"""
        with self.assertRaisesRegex(ValueError, "The dictionary does not represent a ModelService asset."):
            ModelServiceFactory.load_from_dict({"name": "test_model_service", "configs": {"foo": "bar"}})

        with self.assertRaisesRegex(ValueError, "Missing 'inference_api_format' key in the dictionary"):
            ModelServiceFactory.load_from_dict(
                {"name": "test_model_service", "type": "ModelService", "configs": {"foo": "bar"}}
            )

        with self.assertRaisesRegex(ValueError, "Unsupported inference_api_format: unsupported_format"):
            ModelServiceFactory.load_from_dict(
                {
                    "name": "test_model_service",
                    "type": "ModelService",
                    "configs": {"inference_api_format": "unsupported_format"},
                }
            )

    @patch("ravelin.assets.asset_factory.ModelServiceFactory.load_from_reference")
    def test_load_model_service_from_local(self, mock_load_from_reference):
        """Test loading model service from local storage"""
        ModelServiceFactory.load_from_local("test_model_service")
        mock_load_from_reference.assert_called_once_with(AssetReference("test_model_service", source="local"))

    @patch("ravelin.assets.asset_factory.resolve_asset")
    def test_load_model_service_from_reference(self, mock_resolve_asset):
        """Test loading model service from asset reference"""
        mock_resolve_asset.return_value = {
            "name": "test_model_service",
            "type": "ModelService",
            "configs": {"inference_api_format": "test_format"},
        }
        model_service = ModelServiceFactory.load_from_reference(
            AssetReference(name="test_model_service", source="local")
        )

        self.assertIsInstance(model_service, TestModelService)


class TestToolServiceFactory(unittest.TestCase):
    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.tool_services._base.TOOL_SERVICE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Now register our mock class using the decorator function
        register_tool_service("test_format")(TestToolService)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_create_tool_service_with_supported_format(self):
        """Test creating tool service with supported format"""
        tool_service = ToolServiceFactory.create("test_format", name="test_tool_service", uri="test_url")
        self.assertIsInstance(tool_service, TestToolService)

    def test_create_tool_service_with_unsupported_format(self):
        """Test creating tool service with unsupported format raises error"""
        with self.assertRaisesRegex(ValueError, "Unsupported call_api_format: unsupported_format"):
            ToolServiceFactory.create("unsupported_format", name="test_tool_service", uri="test_url")

    def test_load_tool_service_from_dict_with_supported_format(self):
        """Test loading tool service from dictionary with supported format"""
        tool_service = ToolServiceFactory.load_from_dict(
            {
                "name": "test_tool_service",
                "type": "ToolService",
                "configs": {"call_api_format": "test_format", "uri": "test_url"},
                "secrets": {"api_key": Secret.from_token("test_key")},
            }
        )
        self.assertIsInstance(tool_service, TestToolService)
        self.assertEqual("test_url", tool_service.uri)

    def test_load_tool_service_from_dict_with_unsupported_format(self):
        """Test loading tool service from dictionary with unsupported format raises error"""
        with self.assertRaisesRegex(ValueError, "The dictionary does not represent a ToolService asset."):
            ToolServiceFactory.load_from_dict({"name": "test_tool_service", "configs": {"foo": "bar"}})

        with self.assertRaisesRegex(ValueError, "Missing 'call_api_format' key in the dictionary"):
            ToolServiceFactory.load_from_dict(
                {"name": "test_tool_service", "type": "ToolService", "configs": {"foo": "bar", "uri": "test_url"}}
            )

        with self.assertRaisesRegex(ValueError, "Unsupported call_api_format: unsupported_format"):
            ToolServiceFactory.load_from_dict(
                {
                    "name": "test_tool_service",
                    "type": "ToolService",
                    "configs": {"call_api_format": "unsupported_format", "uri": "test_url"},
                }
            )

    @patch("ravelin.assets.asset_factory.ToolServiceFactory.load_from_reference")
    def test_load_tool_service_from_local(self, mock_load_from_reference):
        """Test loading tool service from local storage"""
        ToolServiceFactory.load_from_local("test_tool_service")
        mock_load_from_reference.assert_called_once_with(AssetReference("test_tool_service", source="local"))

    @patch("ravelin.assets.asset_factory.resolve_asset")
    def test_load_tool_service_from_reference(self, mock_resolve_asset):
        """Test loading tool service from asset reference"""
        mock_resolve_asset.return_value = {
            "name": "test_tool_service",
            "type": "ToolService",
            "configs": {"call_api_format": "test_format", "uri": "test_url"},
        }
        model_service = ToolServiceFactory.load_from_reference(
            AssetReference(name="test_model_service", source="local")
        )

        self.assertIsInstance(model_service, TestToolService)


class TestVectorDatabaseFactory(unittest.TestCase):
    def setUp(self):
        # Create a fresh registry for each test
        self.registry = {}

        # Start the patch with an empty registry
        self.registry_patcher = patch(
            "ravelin.assets.vector_databases._base.VECTOR_DATABASE_REGISTRY._registry",
            self.registry,
        )
        self.mock_registry = self.registry_patcher.start()

        # Now register our mock class using the decorator function
        register_vector_database("test_type")(TestVectorDatabase)

    def tearDown(self):
        # Stop the patch after each test
        self.registry_patcher.stop()

    def test_create_vector_database_with_supported_type(self):
        """Test creating vector database with supported type"""
        vector_database = VectorDatabaseFactory.create(
            "test_type",
            name="test_vector_database",
            collection_name="test_collection",
            host="localhost",
            port=19530,
            embedding_fields={
                "field1": {
                    "from_asset": {"name": "abc", "source": "local"},
                    "embedder_kwargs": {"foo": "bar"},
                    "search_kwargs": {"foo": "baz"},
                }
            },
            content_field="abc",
        )
        self.assertIsInstance(vector_database, TestVectorDatabase)

    def test_create_vector_database_with_unsupported_type(self):
        """Test creating vector database with unsupported type raises error"""
        with self.assertRaisesRegex(ValueError, "Unsupported vector_db_type: unsupported_type"):
            VectorDatabaseFactory.create(
                "unsupported_type",
                collection_name="test_collection",
                host="localhost",
                port=19530,
                embedding_fields={},
                content_field="content",
            )

    def test_load_vector_database_from_dict_with_supported_type(self):
        """Test loading vector database from dictionary with supported type"""
        vector_database = VectorDatabaseFactory.load_from_dict(
            {
                "name": "test_vector_database",
                "type": "VectorDatabase",
                "configs": {
                    "vector_db_type": "test_type",
                    "collection_name": "test_collection",
                    "host": "localhost",
                    "port": 19530,
                    "embedding_fields": {
                        "field1": {
                            "from_asset": {"name": "abc", "source": "local"},
                            "embedder_kwargs": {"foo": "bar"},
                            "search_kwargs": {"foo": "baz"},
                        }
                    },
                    "content_field": "content",
                    "meta_fields": ["123", "456"],
                },
                "secrets": {},
            }
        )
        self.assertIsInstance(vector_database, TestVectorDatabase)

    def test_load_vector_database_from_dict_with_unsupported_type(self):
        """Test loading vector database from dictionary with unsupported type raises error"""
        with self.assertRaisesRegex(ValueError, "The dictionary does not represent a VectorDatabase asset."):
            VectorDatabaseFactory.load_from_dict({"name": "test_vector_database", "configs": {"foo": "bar"}})

        with self.assertRaisesRegex(ValueError, "Missing 'vector_db_type' key in the dictionary"):
            VectorDatabaseFactory.load_from_dict(
                {
                    "name": "test_vector_database",
                    "type": "VectorDatabase",
                    "configs": {"foo": "bar", "collection_name": "test_collection"},
                }
            )

        with self.assertRaisesRegex(ValueError, "Unsupported vector_db_type: unsupported_type"):
            VectorDatabaseFactory.load_from_dict(
                {
                    "name": "test_vector_database",
                    "type": "VectorDatabase",
                    "configs": {
                        "vector_db_type": "unsupported_type",
                        "collection_name": "test_collection",
                        "content_field": "content",
                        "embedding_fields": {},
                        "host": "localhost",
                        "port": 19530,
                    },
                }
            )

    @patch("ravelin.assets.asset_factory.VectorDatabaseFactory.load_from_reference")
    def test_load_vector_database_from_local(self, mock_load_from_reference):
        """Test loading vector database from local storage"""
        VectorDatabaseFactory.load_from_local("test_vector_database")
        mock_load_from_reference.assert_called_once_with(AssetReference("test_vector_database", source="local"))

    @patch("ravelin.assets.asset_factory.resolve_asset")
    def test_load_vector_database_from_reference(self, mock_resolve_asset):
        """Test loading vector database from asset reference"""
        mock_resolve_asset.return_value = {
            "name": "test_vector_database",
            "type": "VectorDatabase",
            "configs": {
                "vector_db_type": "test_type",
                "collection_name": "test_collection",
                "host": "localhost",
                "port": 19530,
                "embedding_fields": {
                    "field1": {
                        "from_asset": {"name": "abc", "source": "local"},
                        "embedder_kwargs": {"foo": "bar"},
                        "search_kwargs": {"foo": "baz"},
                    }
                },
                "content_field": "content",
                "meta_fields": ["123", "456"],
            },
            "secrets": {},
        }
        vector_database = VectorDatabaseFactory.load_from_reference(
            AssetReference(name="test_vector_database", source="local")
        )

        self.assertIsInstance(vector_database, TestVectorDatabase)


class TestAssetFactory(unittest.TestCase):
    """Tests for asset factory"""

    def setUp(self):
        # Create fresh registries for each test
        self.model_registry = {}
        self.tool_registry = {}
        self.vector_db_registry = {}

        # Start the patches with empty registries
        self.model_registry_patcher = patch(
            "ravelin.assets.model_services._base.MODEL_SERVICE_REGISTRY._registry",
            self.model_registry,
        )
        self.tool_registry_patcher = patch(
            "ravelin.assets.tool_services._base.TOOL_SERVICE_REGISTRY._registry",
            self.tool_registry,
        )
        self.vector_db_registry_patcher = patch(
            "ravelin.assets.vector_databases._base.VECTOR_DATABASE_REGISTRY._registry",
            self.vector_db_registry,
        )

        self.mock_model_registry = self.model_registry_patcher.start()
        self.mock_tool_registry = self.tool_registry_patcher.start()
        self.mock_vector_db_registry = self.vector_db_registry_patcher.start()

        # Register our mock classes using the decorator functions
        register_model_service("test_format")(TestModelService)
        register_tool_service("test_format")(TestToolService)
        register_vector_database("test_type")(TestVectorDatabase)

    def tearDown(self):
        # Stop the patches after each test
        self.model_registry_patcher.stop()
        self.tool_registry_patcher.stop()
        self.vector_db_registry_patcher.stop()

    def test_create_model_service_asset(self):
        """Test creating model service asset"""
        asset = AssetFactory.create("ModelService", inference_api_format="test_format", name="test_asset")
        self.assertIsInstance(asset, TestModelService)

    def test_create_tool_service_asset(self):
        """Test creating tool service asset"""
        asset = AssetFactory.create("ToolService", call_api_format="test_format", name="test_asset", uri="test_url")
        self.assertIsInstance(asset, TestToolService)

    def test_create_vector_database_asset(self):
        """Test creating vector database asset"""
        asset = AssetFactory.create(
            "VectorDatabase",
            vector_db_type="test_type",
            name="test_asset",
            collection_name="test_collection",
            host="localhost",
            port=19530,
            embedding_fields={
                "field1": {
                    "from_asset": {"name": "abc", "source": "local"},
                    "embedder_kwargs": {"foo": "bar"},
                    "search_kwargs": {"foo": "baz"},
                }
            },
            content_field="content",
        )
        self.assertIsInstance(asset, TestVectorDatabase)

    def test_create_asset_with_unsupported_type(self):
        """Test creating asset with unsupported type raises error"""
        with self.assertRaisesRegex(ValueError, "Unsupported asset_type: UnsupportedType"):
            AssetFactory.create("UnsupportedType")

    @patch("ravelin.assets.asset_factory.AssetFactory.load_from_reference")
    def test_load_asset_from_local(self, mock_load_from_reference):
        """Test loading asset from local storage"""
        AssetFactory.load_from_local("test_asset")
        mock_load_from_reference.assert_called_once_with(AssetReference("test_asset", source="local"))

    @patch("ravelin.assets.asset_factory.resolve_asset")
    @patch("ravelin.assets.asset_factory.AssetFactory.load_from_dict")
    def test_load_asset_from_reference(self, mock_load_from_dict, mock_resolve_asset):
        """Test loading asset from asset reference"""
        mock_resolve_asset.return_value = {"test": "data"}
        AssetFactory.load_from_reference(AssetReference(name="test_asset", source="local"))
        mock_load_from_dict.assert_called_once_with({"test": "data"})


if __name__ == "__main__":
    unittest.main()
