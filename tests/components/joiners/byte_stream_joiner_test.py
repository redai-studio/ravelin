import unittest

from ravelin.components.joiners.byte_stream_joiner import ByteStreamJoiner
from ravelin.dataclasses.byte_stream import ByteStream


class TestByteStreamJoiner(unittest.TestCase):
    def setUp(self):
        self.joiner = ByteStreamJoiner()

    def test_run(self):
        results = self.joiner.run(byte_streams=[[ByteStream(data=b"hello")], [ByteStream(data=b"world")]])
        self.assertListEqual([ByteStream(data=b"hello"), ByteStream(data=b"world")], results["byte_streams"])


if __name__ == "__main__":
    unittest.main()
