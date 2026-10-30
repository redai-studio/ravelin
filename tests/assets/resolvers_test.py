import unittest
from unittest.mock import patch

from haystack.utils import Secret
from haystack.utils.auth import EnvVarSecret, TokenSecret

from ravelin.assets.resolvers import (
    LocalAssetResolver,
    normalize_asset_secrets,
    register_asset_resolver,
    resolve_asset,
)
from ravelin.assets.asset_factory import ModelServiceFactory
from ravelin.dataclasses import AssetReference


class StubResolver:
    def __init__(self, data):
        self.data = data

    def resolve(self, reference):
        return self.data


class TestAssetResolvers(unittest.TestCase):
    def setUp(self):
        self.registry_patcher = patch(
            "ravelin.assets.resolvers._ASSET_RESOLVERS",
            {"local": LocalAssetResolver()},
        )
        self.registry_patcher.start()
        self.addCleanup(self.registry_patcher.stop)

    def test_registers_and_resolves_custom_source(self):
        data = {"name": "service", "type": "model-service", "configs": {}, "secrets": {}}
        register_asset_resolver("company-registry", StubResolver(data))

        resolved = resolve_asset(AssetReference("service", source="company-registry"))

        self.assertEqual(data, resolved)

    def test_rejects_duplicate_registration(self):
        register_asset_resolver("company-registry", StubResolver({}))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_asset_resolver("company-registry", StubResolver({}))

    def test_rejects_reserved_local_source(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            register_asset_resolver("local", StubResolver({}))

    def test_rejects_invalid_registration(self):
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            register_asset_resolver("", StubResolver({}))
        with self.assertRaisesRegex(TypeError, "callable resolve"):
            register_asset_resolver("invalid", object())

    def test_rejects_unregistered_source(self):
        with self.assertRaisesRegex(ValueError, "No asset resolver is registered"):
            resolve_asset(AssetReference("service", source="missing"))

    @patch("ravelin.assets.resolvers.get_local_asset")
    def test_local_resolver_deserializes_environment_secret(self, mock_get_local_asset):
        mock_get_local_asset.return_value = {
            "name": "service",
            "secrets": {
                "api_key": {"type": "env_var", "env_vars": ["SERVICE_API_KEY"], "strict": True},
            },
        }

        resolved = resolve_asset(AssetReference("service"))

        self.assertIsInstance(resolved["secrets"]["api_key"], EnvVarSecret)

    @patch("ravelin.assets.resolvers.get_local_asset")
    def test_local_resolver_rejects_plaintext_secret(self, mock_get_local_asset):
        mock_get_local_asset.return_value = {"name": "service", "secrets": {"api_key": "not-logged"}}
        with self.assertRaisesRegex(TypeError, "api_key"):
            resolve_asset(AssetReference("service"))

    @patch("ravelin.assets.resolvers.get_local_asset")
    def test_local_resolver_rejects_token_secret(self, mock_get_local_asset):
        mock_get_local_asset.return_value = {
            "name": "service",
            "secrets": {"api_key": Secret.from_token("not-logged")},
        }
        with self.assertRaisesRegex(ValueError, "environment-variable reference"):
            resolve_asset(AssetReference("service"))

    def test_trusted_resolver_can_normalize_plaintext_to_token_secret(self):
        normalized = normalize_asset_secrets(
            {"secrets": {"api_key": "not-logged"}},
            allow_plaintext=True,
        )
        self.assertIsInstance(normalized["secrets"]["api_key"], TokenSecret)

    def test_secret_normalization_validates_mapping_shape(self):
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            normalize_asset_secrets({"secrets": ["invalid"]})
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            normalize_asset_secrets({"secrets": []})
        with self.assertRaisesRegex(TypeError, "names must be strings"):
            normalize_asset_secrets({"secrets": {1: None}})
        with self.assertRaisesRegex(TypeError, "Haystack Secret"):
            normalize_asset_secrets({"secrets": {"api_key": 123}})

    def test_factory_retains_reference_but_cannot_serialize_resolved_token(self):
        reference = AssetReference("service", source="company-registry", version="v1")
        data = normalize_asset_secrets(
            {
                "name": "service",
                "type": "model-service",
                "configs": {
                    "inference_api_format": "openai-chat",
                    "base_url": "https://example.com",
                },
                "secrets": {"api_key": "not-logged"},
            },
            allow_plaintext=True,
        )
        register_asset_resolver("company-registry", StubResolver(data))

        asset = ModelServiceFactory.load_from_reference(reference)

        self.assertEqual(reference, asset.reference)
        self.assertIsInstance(asset.secrets["api_key"], TokenSecret)
        with self.assertRaisesRegex(ValueError, "Cannot serialize token-based secret"):
            asset.to_dict()


if __name__ == "__main__":
    unittest.main()
