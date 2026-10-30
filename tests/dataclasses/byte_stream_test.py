import unittest
from base64 import b64encode
from pathlib import Path
from unittest.mock import mock_open, patch

from ravelin.dataclasses.byte_stream import ByteStream


class TestByteStream(unittest.TestCase):
    def setUp(self):
        self.test_data = b"test data"
        self.test_meta = {"key": "value"}
        self.test_mime = "text/plain"
        self.byte_stream = ByteStream(data=self.test_data, meta=self.test_meta, mime_type=self.test_mime)

    def test_init(self):
        self.assertEqual(self.test_data, self.byte_stream.data)
        self.assertEqual(self.test_meta, self.byte_stream.meta)
        self.assertEqual(self.test_mime, self.byte_stream.mime_type)

    def test_type_property(self):
        self.assertEqual("text", self.byte_stream.type)
        stream_without_mime = ByteStream(data=self.test_data)
        self.assertIsNone(stream_without_mime.type)

    def test_subtype_property(self):
        self.assertEqual("plain", self.byte_stream.subtype)
        stream_without_mime = ByteStream(data=self.test_data)
        self.assertIsNone(stream_without_mime.subtype)

    @patch("builtins.open", new_callable=mock_open)
    def test_to_file(self, mock_file):
        path = Path("test.txt")
        self.byte_stream.to_file(path)
        mock_file.assert_called_once_with(path, "wb")
        mock_file().write.assert_called_once_with(self.test_data)

    def test_from_string(self):
        text = "Hello, World!"
        byte_stream = ByteStream.from_string(text, mime_type="text/plain")
        self.assertEqual(text.encode("utf-8"), byte_stream.data)
        self.assertEqual("text/plain", byte_stream.mime_type)

    def test_to_string(self):
        byte_stream = ByteStream(data=b"Hello, World!")
        self.assertEqual("Hello, World!", byte_stream.to_string())

    def test_from_base64(self):
        base64_string = b64encode(b"test data").decode("utf-8")
        byte_stream = ByteStream.from_base64(base64_string, mime_type="text/plain")
        self.assertEqual(b"test data", byte_stream.data)
        self.assertEqual("text/plain", byte_stream.mime_type)

    def test_to_base64(self):
        expected = b64encode(self.test_data).decode("utf-8")
        self.assertEqual(expected, self.byte_stream.to_base64())

    def test_from_dict(self):
        data = {
            "data": b64encode(self.test_data).decode("utf-8"),
            "meta": self.test_meta,
            "mime_type": self.test_mime,
        }
        byte_stream = ByteStream.from_dict(data)
        self.assertEqual(self.test_data, byte_stream.data)
        self.assertEqual(self.test_meta, byte_stream.meta)
        self.assertEqual(self.test_mime, byte_stream.mime_type)

    def test_to_dict(self):
        expected = {
            "data": b64encode(self.test_data).decode("utf-8"),
            "meta": self.test_meta,
            "mime_type": self.test_mime,
        }
        self.assertEqual(expected, self.byte_stream.to_dict())

    def test_to_trace_dict(self):
        expected = {
            "data": f"Base 64 data ({len(b64encode(self.test_data).decode('utf-8'))} bytes)",
            "meta": self.test_meta,
            "mime_type": self.test_mime,
        }
        self.assertEqual(expected, self.byte_stream._to_trace_dict())


if __name__ == "__main__":
    unittest.main()
