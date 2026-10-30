import os
import unittest
from unittest.mock import patch, mock_open, Mock

from haystack.utils import Secret
from haystack.utils.auth import EnvVarSecret

from ravelin.assets import Asset
from ravelin.assets._base import resolve_secret


class TestAsset(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case. This method will be called before each test.
        """
        self.asset = Asset(
            name="test",
            configs={"config1": "value1"},
            secrets={"secret1": Secret.from_env_var("TEST_SECRET")},
        )

    def test_reference(self):
        with self.assertRaisesRegex(
            ValueError, 'The asset "test" is not loaded from a reference and thus cannot be serialized.'
        ):
            _ = self.asset.reference

        mock_asset_reference = Mock()
        self.asset._reference = mock_asset_reference
        self.assertEqual(mock_asset_reference, self.asset.reference)

    def test_to_dict(self):
        """
        Test the to_dict method.
        """
        expected_dict = {
            "name": "test",
            "type": "not-set",
            "configs": {"config1": "value1"},
            "secrets": {"secret1": {"type": "env_var", "env_vars": ["TEST_SECRET"], "strict": True}},
        }
        self.assertEqual(expected_dict, self.asset.to_dict())

    def test_from_dict(self):
        """
        Test the from_dict method.
        """
        data = {
            "name": "test",
            "type": "not-set",
            "configs": {"config1": "value1"},
            "secrets": {"secret1": {"type": "env_var", "env_vars": ["TEST_SECRET"], "strict": True}},
        }
        asset = Asset.from_dict(data)
        self.assertEqual("test", asset.name)
        self.assertEqual("not-set", asset.ASSET_TYPE)
        self.assertEqual({"config1": "value1"}, asset.configs)
        self.assertIsInstance(asset.secrets["secret1"], EnvVarSecret)

    @patch("ravelin.assets._base.AssetReference")
    @patch("ravelin.assets._base.get_local_asset_path")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_to_local_storage(self, mock_file_open, mock_get_local_asset_path, mock_asset_reference):
        mock_get_local_asset_path.return_value.exists.return_value = False
        self.asset.save_to_local_storage()
        mock_asset_reference.assert_called_once_with(name="test", source="local")
        mock_file_open.assert_called_once()
        expected = (
            '{"name": "test", "type": "not-set", "configs": {"config1": "value1"}, '
            '"secrets": {"secret1": {"type": "env_var", "env_vars": ["TEST_SECRET"], "strict": true}}}'
        )
        results = "".join([call.args[0] for call in mock_file_open().write.call_args_list])
        self.assertEqual(expected, results)
        self.assertEqual(mock_asset_reference.return_value, self.asset.reference)

    @patch("ravelin.assets._base.logger")
    @patch("ravelin.assets._base.get_local_asset_path")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_to_local_storage_exists(self, mock_file_open, mock_get_local_asset_path, mock_logger):
        mock_get_local_asset_path.return_value.exists.return_value = True
        self.asset.save_to_local_storage()

        mock_logger.warning.assert_called_once()
        mock_file_open.assert_not_called()

    @patch("ravelin.assets._base.get_local_asset_path")
    @patch("builtins.open", new_callable=mock_open)
    def test_token_secret_fails_before_opening_local_file(self, mock_file_open, mock_get_local_asset_path):
        mock_get_local_asset_path.return_value.exists.return_value = False
        asset = Asset(name="test", secrets={"api_key": Secret.from_token("not-logged")})

        with self.assertRaisesRegex(ValueError, "Cannot serialize token-based secret"):
            asset.save_to_local_storage()

        mock_file_open.assert_not_called()
        with self.assertRaisesRegex(ValueError, "not loaded from a reference"):
            _ = asset.reference


class TestResolveSecret(unittest.TestCase):
    def test_resolve_none(self):
        self.assertIsNone(resolve_secret(None))

    def test_rejects_plain_string(self):
        with self.assertRaisesRegex(TypeError, "Haystack Secret"):
            resolve_secret("plain-token")

    def test_resolve_env_var_secret(self):
        secret = Secret.from_env_var("TEST_SECRET_ENV")
        with patch.dict(os.environ, {"TEST_SECRET_ENV": "resolved-value"}):
            self.assertEqual("resolved-value", resolve_secret(secret))

    def test_resolve_token_secret(self):
        self.assertEqual("sk-token", resolve_secret(Secret.from_token("sk-token")))

    def test_resolve_strict_env_var_unset_raises(self):
        secret = Secret.from_env_var("TEST_SECRET_UNSET")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_secret(secret)


class TestAssetSecretSerialization(unittest.TestCase):
    """Assets only retain Haystack ``Secret`` objects and serialize safe references."""

    def test_plain_string_secret_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "api_key"):
            Asset(name="test", secrets={"api_key": "abc"})

    def test_env_var_secret_serializes_to_reference(self):
        asset = Asset(name="test", secrets={"api_key": Secret.from_env_var("MY_KEY")})
        serialized = asset.to_dict()
        # The value itself is never persisted; only an env-var reference is written.
        self.assertEqual(
            {"type": "env_var", "env_vars": ["MY_KEY"], "strict": True},
            serialized["secrets"]["api_key"],
        )

    def test_env_var_secret_round_trip(self):
        asset = Asset(name="test", secrets={"api_key": Secret.from_env_var("MY_KEY")})
        restored = Asset.from_dict(asset.to_dict())
        self.assertIsInstance(restored.secrets["api_key"], EnvVarSecret)
        with patch.dict(os.environ, {"MY_KEY": "env-value"}):
            self.assertEqual("env-value", resolve_secret(restored.secrets["api_key"]))

    def test_token_secret_raises_on_to_dict(self):
        asset = Asset(name="test", secrets={"api_key": Secret.from_token("sk-token")})
        with self.assertRaises(ValueError):
            asset.to_dict()


if __name__ == "__main__":
    unittest.main()
