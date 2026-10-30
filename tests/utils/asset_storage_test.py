import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, mock_open

from ravelin.assets import Asset
from ravelin.assets.resolvers import LocalAssetResolver
from ravelin.dataclasses import AssetReference
from ravelin.utils import asset_storage


class TestAssetStorage(unittest.TestCase):
    def setUp(self):
        self.asset_name = "test_asset"
        self.asset_data = {
            "name": "test_asset",
            "type": "model-service",
            "configs": {"inference_api_format": "test_format"},
        }
        self.asset_path = "/path/to/assets"

    def test_set_local_asset_storage_path(self):
        asset_storage.set_local_asset_storage_path(self.asset_path)
        self.assertEqual(os.environ[asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH], self.asset_path)

    def test_get_local_asset_storage_path(self):
        os.environ[asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH] = self.asset_path
        self.assertEqual(asset_storage.get_local_asset_storage_path(), Path(self.asset_path))

    def test_get_local_asset_storage_path_not_set(self):
        if asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH in os.environ:
            del os.environ[asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH]
        with self.assertRaises(ValueError):
            asset_storage.get_local_asset_storage_path()

    def test_get_local_asset_path(self):
        os.environ[asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH] = self.asset_path
        expected_path = Path(self.asset_path) / f"{self.asset_name}.json"
        self.assertEqual(asset_storage.get_local_asset_path(self.asset_name), expected_path)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_get_local_asset(self, mock_file_open, _):
        mock_file_open.return_value.read.return_value = json.dumps(self.asset_data)
        os.environ[asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH] = self.asset_path
        self.assertEqual(asset_storage.get_local_asset(self.asset_name), self.asset_data)
        mock_file_open.assert_called_once_with(Path(self.asset_path) / f"{self.asset_name}.json", "r", encoding="utf-8")

    @patch("pathlib.Path.exists", return_value=False)
    def test_get_local_asset_not_found(self, _):
        os.environ[asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH] = self.asset_path
        with self.assertRaises(FileNotFoundError):
            asset_storage.get_local_asset(self.asset_name)


class TestLocalAssetContainment(unittest.TestCase):
    """Exercise local reads and writes against real paths and symlinks."""

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.parent = Path(directory.name).resolve()
        self.root = self.parent / "assets"
        self.root.mkdir()
        environment = patch.dict(os.environ, {asset_storage.ENV_LOCAL_ASSET_STORAGE_PATH: str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)

    def assert_rejected_for_read_and_write(self, name):
        with self.assertRaises(ValueError):
            LocalAssetResolver().resolve(AssetReference(name=name))
        with self.assertRaises(ValueError):
            Asset(name=name).save_to_local_storage(allow_overwrite=True)

    def test_nested_asset_round_trip(self):
        (self.root / "models").mkdir()
        asset = Asset(name="models/primary-chat-model", configs={"key": "value"})

        asset.save_to_local_storage()
        restored = LocalAssetResolver().resolve(asset.reference)

        self.assertEqual(asset.to_dict(), restored)
        self.assertEqual(
            self.root / "models/primary-chat-model.json",
            asset_storage.get_local_asset_path(asset.name),
        )

    def test_rejects_invalid_local_names(self):
        names = [
            None,
            42,
            "",
            " ",
            ".",
            "..",
            "../outside",
            "models/../outside",
            "models/./primary",
            "models//primary",
            "models/",
            str(self.parent / "outside"),
            r"C:\outside",
            "C:/outside",
            "C:outside",
            r"\\server\share\outside",
            r"models\..\outside",
            "primary:alternate-stream",
            "primary\0ignored",
        ]
        for name in names:
            with self.subTest(name=name):
                self.assert_rejected_for_read_and_write(name)

    def test_traversal_cannot_read_or_overwrite_existing_file(self):
        outside = self.parent / "outside.json"
        original = '{"name": "outside", "secrets": {}}'
        outside.write_text(original, encoding="utf-8")

        self.assert_rejected_for_read_and_write("../outside")

        self.assertEqual(original, outside.read_text(encoding="utf-8"))

    def test_directory_symlink_cannot_escape_for_reads_or_new_writes(self):
        outside = self.parent / "assets-private"
        outside.mkdir()
        (outside / "existing.json").write_text('{"name": "existing"}', encoding="utf-8")
        (self.root / "linked").symlink_to(outside, target_is_directory=True)

        self.assert_rejected_for_read_and_write("linked/existing")
        self.assert_rejected_for_read_and_write("linked/new")

        self.assertFalse((outside / "new.json").exists())
        self.assertEqual('{"name": "existing"}', (outside / "existing.json").read_text(encoding="utf-8"))

    def test_file_symlink_cannot_escape_even_when_target_is_missing(self):
        outside = self.parent / "outside.json"
        link = self.root / "linked.json"
        link.symlink_to(outside)
        self.assert_rejected_for_read_and_write("linked")
        self.assertFalse(outside.exists())

        outside.write_text('{"name": "outside"}', encoding="utf-8")
        self.assert_rejected_for_read_and_write("linked")
        self.assertEqual('{"name": "outside"}', outside.read_text(encoding="utf-8"))

    def test_symlink_within_root_remains_usable(self):
        Asset(name="primary", configs={"key": "value"}).save_to_local_storage()
        (self.root / "alias.json").symlink_to(self.root / "primary.json")

        restored = LocalAssetResolver().resolve(AssetReference(name="alias"))

        self.assertEqual({"key": "value"}, restored["configs"])
        self.assertEqual(self.root / "primary.json", asset_storage.get_local_asset_path("alias"))

    def test_configured_root_can_be_a_symlink(self):
        alias = self.parent / "asset-root"
        alias.symlink_to(self.root, target_is_directory=True)
        asset_storage.set_local_asset_storage_path(alias)

        Asset(name="primary").save_to_local_storage()

        self.assertEqual(self.root / "primary.json", asset_storage.get_local_asset_path("primary"))


if __name__ == "__main__":
    unittest.main()
