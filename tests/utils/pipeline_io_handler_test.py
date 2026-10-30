import unittest
from io import StringIO
from unittest.mock import MagicMock

from haystack import Pipeline

from ravelin.utils.pipeline_io_handler import PipelineIOHandler, InputSpec, OutputSpec


class TestPipelineIOHandler(unittest.TestCase):
    def setUp(self):
        self.input_specs = {
            "query": {
                "type": "str",
                "connections": ["retriever.query", "builder.query"],
                "required": True,
                "description": "The query to be processed.",
            },
            "user_context": {
                "type": "typing.Any",
                "connections": ["builder.user_context"],
                "required": False,
                "description": None,
            },
            "temperature": {
                "type": "float",
                "connections": ["generator.temperature"],
                "required": False,
                "description": None,
            },
        }
        self.output_specs = {
            "replies": {
                "type": "typing.List[str]",
                "connection": "generator.replies",
                "required": True,
                "description": "The replies.",
            },
            "context": {
                "type": "typing.Any",
                "connection": "retriever.documents",
                "required": False,
                "description": None,
            },
        }
        self.handler = PipelineIOHandler(inputs=self.input_specs, outputs=self.output_specs)

    def test_from_and_to_dict(self):
        handler_dict = {
            "inputs": {
                "query": InputSpec(
                    type=str, connections=["retriever.query", "builder.query"], description="The query to be processed."
                ),
                "user_context": {"connections": ["builder.user_context"], "required": False},
                "temperature": {"type": "float", "connections": ["generator.temperature"], "required": False},
            },
            "outputs": {
                "replies": {
                    "type": "typing.List[str]",
                    "connection": "generator.replies",
                    "description": "The replies.",
                },
                "context": OutputSpec(connection="retriever.documents", required=False),
            },
        }
        handler = PipelineIOHandler.from_dict(handler_dict)
        handler_dict = handler.to_dict()
        self.assertEqual(self.input_specs, handler_dict["inputs"])
        self.assertEqual(self.output_specs, handler_dict["outputs"])

    def test_dumps_and_loads(self):
        handler_str = self.handler.dumps()
        loaded_handler = PipelineIOHandler.loads(handler_str)
        self.assertEqual(self.input_specs, {name: spec.to_dict() for name, spec in loaded_handler.input_specs.items()})
        self.assertEqual(
            self.output_specs, {name: spec.to_dict() for name, spec in loaded_handler.output_specs.items()}
        )

    def test_dump_and_load(self):
        with StringIO() as fp:
            self.handler.dump(fp)
            fp.seek(0)
            loaded_handler = PipelineIOHandler.load(fp)
        self.assertEqual(self.input_specs, {name: spec.to_dict() for name, spec in loaded_handler.input_specs.items()})
        self.assertEqual(
            self.output_specs, {name: spec.to_dict() for name, spec in loaded_handler.output_specs.items()}
        )

    def test_parse_input_default(self):
        parsed_input = self.handler.parse_input({"query": "Hello!"})
        self.assertIn("retriever", parsed_input)
        self.assertIn("builder", parsed_input)
        self.assertEqual("Hello!", parsed_input["retriever"]["query"])
        self.assertEqual("Hello!", parsed_input["builder"]["query"])
        self.assertNotIn("user_context", parsed_input["builder"])
        self.assertNotIn("temperature", parsed_input["generator"])

    def test_parse_input_full(self):
        parsed_input = self.handler.parse_input({"query": "Hello!", "user_context": "World!", "temperature": 0.6})
        self.assertIn("retriever", parsed_input)
        self.assertIn("builder", parsed_input)
        self.assertEqual("Hello!", parsed_input["retriever"]["query"])
        self.assertEqual("Hello!", parsed_input["builder"]["query"])
        self.assertEqual("World!", parsed_input["builder"]["user_context"])
        self.assertEqual(0.6, parsed_input["generator"]["temperature"])

    def test_parse_input_missing_required(self):
        with self.assertRaisesRegex(ValueError, 'Input variable "query" is required but not provided.'):
            self.handler.parse_input({"user_context": "World!"})

    def test_get_output_components(self):
        output_components = self.handler.get_output_components()
        self.assertEqual({"generator", "retriever"}, output_components)

    def test_get_output_components_missing_connection(self):
        self.output_specs["context"]["connection"] = ""
        handler = PipelineIOHandler(inputs=self.input_specs, outputs=self.output_specs)
        with self.assertRaisesRegex(ValueError, 'No output connection found for output variable "context": '):
            handler.get_output_components()

    def test_parse_output(self):
        raw_output_data = {"retriever": {"documents": "Foo"}, "generator": {"replies": "Bar"}}
        parsed_output = self.handler.parse_output(raw_output_data)
        self.assertEqual({"replies": "Bar", "context": "Foo"}, parsed_output)

    def test_parse_output_missing_connection(self):
        self.output_specs["context"]["connection"] = ""
        handler = PipelineIOHandler(inputs=self.input_specs, outputs=self.output_specs)
        raw_output_data = {"retriever": {"documents": "Foo"}, "generator": {"replies": "Bar"}}
        with self.assertRaisesRegex(ValueError, 'No output connection found for output variable "context": '):
            handler.parse_output(raw_output_data)

    def test_parse_output_missing_component_output(self):
        raw_output_data = {"retriever": {"documents": "Foo"}}
        with self.assertRaisesRegex(ValueError, 'No output value found for required output variable "replies".'):
            self.handler.parse_output(raw_output_data)

    def test_run_pipeline(self):
        mock_pipeline = MagicMock(spec=Pipeline)
        data = {"query": "Hello!"}
        mock_pipeline.run.return_value = {"retriever": {"documents": "Foo"}, "generator": {"replies": "Bar"}}
        output = self.handler.run_pipeline(mock_pipeline, data)
        self.assertEqual({"replies": "Bar", "context": "Foo"}, output)
        mock_pipeline.run.assert_called_once_with(
            {"retriever": {"query": "Hello!"}, "builder": {"query": "Hello!"}},
            include_outputs_from={"generator", "retriever"},
        )

    def test_get_input_type_hints(self):
        """Test that get_input_type_hints returns mapping of input names to types."""
        type_hints = self.handler.get_input_type_hints()
        self.assertEqual(str, type_hints["query"])
        self.assertEqual(float, type_hints["temperature"])

    def test_get_output_type_hints(self):
        """Test that get_output_type_hints returns mapping of output names to types."""
        type_hints = self.handler.get_output_type_hints()
        # The type is stored as string "typing.List[str]", resolved at runtime
        self.assertIn("replies", type_hints)
        self.assertIn("context", type_hints)

    def test_deserialize_request(self):
        """Test that deserialize_request handles type conversion."""
        from ravelin.dataclasses import ChatMessage

        # Create handler with ChatMessage type
        handler = PipelineIOHandler(
            inputs={
                "messages": {
                    "type": list[ChatMessage],
                    "connections": ["component.messages"],
                    "required": True,
                }
            },
            outputs={},
        )

        # Deserialize raw dict into ChatMessage
        raw_data = {
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        }
        result = handler.deserialize_request(raw_data)

        self.assertEqual(len(result["messages"]), 1)
        self.assertIsInstance(result["messages"][0], ChatMessage)
        self.assertEqual(result["messages"][0].text, "Hello")

    def test_serialize_response(self):
        """Test that serialize_response converts objects to dicts."""
        from ravelin.dataclasses import ChatMessage

        # Create handler with ChatMessage output
        handler = PipelineIOHandler(
            inputs={},
            outputs={
                "reply": {
                    "type": ChatMessage,
                    "connection": "component.reply",
                    "required": True,
                }
            },
        )

        # Serialize ChatMessage to dict
        data = {"reply": ChatMessage.from_assistant("Hello!")}
        result = handler.serialize_response(data)

        self.assertIsInstance(result["reply"], dict)
        self.assertIn("__type__", result["reply"])
        self.assertEqual(result["reply"]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
