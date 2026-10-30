import unittest
from importlib import metadata, reload
from unittest.mock import patch

import pytest

import ravelin.version


class TestVersionRetrieval(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # This will be executed before each test method
        self.patcher = patch.object(metadata, "version")
        self.mock_version = self.patcher.start()

        yield  # this is where the testing happens

        # This will be executed after each test method
        self.patcher.stop()

    def test_version_retrieval_when_package_found(self):
        self.mock_version.return_value = "1.0.0"
        reload(ravelin.version)
        assert ravelin.version.__version__ == "1.0.0"

    def test_version_retrieval_when_package_not_found(self):
        self.mock_version.side_effect = metadata.PackageNotFoundError
        reload(ravelin.version)
        assert ravelin.version.__version__ == "main"


if __name__ == "__main__":
    unittest.main()
