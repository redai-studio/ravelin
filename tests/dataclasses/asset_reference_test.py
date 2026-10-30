import unittest

from ravelin.dataclasses.asset_reference import AssetReference


class TestAssetReference(unittest.TestCase):
    def setUp(self):
        self.asset_reference = AssetReference(name="test_asset", source="local")

    def test_initialization(self):
        self.assertEqual("test_asset", self.asset_reference.name)
        self.assertEqual("local", self.asset_reference.source)

    def test_to_dict(self):
        expected_dict = {"name": "test_asset", "source": "local", "version": None}
        self.assertDictEqual(expected_dict, self.asset_reference.to_dict())

    def test_from_dict(self):
        data = {"name": "test_asset", "source": "local"}
        asset_reference = AssetReference.from_dict(data)
        self.assertEqual(self.asset_reference, asset_reference)

    def test_custom_source_round_trip(self):
        reference = AssetReference(name="test_asset", source="company-registry", version="v1")
        self.assertEqual(reference, AssetReference.from_dict(reference.to_dict()))

    def test_rejects_empty_source(self):
        with self.assertRaisesRegex(ValueError, "source must be a non-empty string"):
            AssetReference(name="test_asset", source="")


if __name__ == "__main__":
    unittest.main()
