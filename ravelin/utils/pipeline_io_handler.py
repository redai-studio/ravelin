"""
This module provides the PipelineIOHandler class for handling the inputs and outputs of a pipeline. The handler is used
to parse the pipeline-level inputs and outputs to the component-level inputs and outputs.

It also provides dynamic Pydantic model creation for request/response validation at HTTP boundaries.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, TextIO, Union, Optional

from haystack.core.pipeline.base import parse_connect_string, DEFAULT_MARSHALLER
from haystack.core.pipeline.pipeline import Pipeline
from haystack.marshal import Marshaller
from haystack.utils import deserialize_type, serialize_type
from typing_extensions import Self

from ravelin.core.serialization import deserialize_value, serialize_value


@dataclass
class InputSpec:
    """Specification for a pipeline input."""

    connections: List[str]
    type: Any = Any
    required: bool = True
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an InputSpec instance from a dictionary.

        :param data: A dictionary containing the input specification.
        :return: An InputSpec instance.
        """
        if "type" in data:
            type_ = deserialize_type(data["type"]) if isinstance(data["type"], str) else data["type"]
        else:
            type_ = Any
        return cls(
            type=type_,
            connections=data.get("connections", []),
            required=data.get("required", False),
            description=data.get("description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the InputSpec.

        :return: A dictionary representation of the input specification.
        """
        return {
            "type": serialize_type(self.type),
            "connections": self.connections,
            "required": self.required,
            "description": self.description,
        }


@dataclass
class OutputSpec:
    """Specification for a pipeline output."""

    connection: str
    type: Any = Any
    required: bool = True
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an OutputSpec instance from a dictionary.

        :param data: A dictionary containing the output specification.
        :return: An OutputSpec instance.
        """
        if "type" in data:
            type_ = deserialize_type(data["type"]) if isinstance(data["type"], str) else data["type"]
        else:
            type_ = Any
        return cls(
            type=type_,
            connection=data["connection"],
            required=data.get("required", True),
            description=data.get("description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the OutputSpec.

        :return: A dictionary representation of the output specification.
        """
        return {
            "type": serialize_type(self.type),
            "connection": self.connection,
            "required": self.required,
            "description": self.description,
        }


class PipelineIOHandler:
    """
    A class used to convert between pipeline-level inputs/outputs and component-level inputs/outputs.

    Usage example::

        io_handler = PipelineIOHandler(
            inputs={"query": {"type": str, "connections": ["retriever.query", "builder.query"]}},
            outputs={"replies": {"type": str, "connection": "generator.replies"}},
        )
        component_inputs = io_handler.parse_input({"query": "Hello!"})
        res = pipeline.run(component_inputs, include_outputs_from=io_handler.get_output_components())
        output = io_handler.parse_output(res)

    :param inputs: A dictionary containing the input specifications for the pipeline.
    :param outputs: A dictionary containing the output specifications for the pipeline.
    """

    def __init__(
        self, inputs: Dict[str, Union[InputSpec, Dict[str, Any]]], outputs: Dict[str, Union[OutputSpec, Dict[str, Any]]]
    ):
        self.input_specs = {
            name: spec if isinstance(spec, InputSpec) else InputSpec.from_dict(spec) for name, spec in inputs.items()
        }
        self.output_specs = {
            name: spec if isinstance(spec, OutputSpec) else OutputSpec.from_dict(spec) for name, spec in outputs.items()
        }

    def to_dict(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns a dictionary representation of the PipelineIOHandler.

        :return: A dictionary representation of the handler.
        """
        return {
            "inputs": {name: spec.to_dict() for name, spec in self.input_specs.items()},
            "outputs": {name: spec.to_dict() for name, spec in self.output_specs.items()},
        }

    @classmethod
    def from_dict(cls, data) -> Self:
        """
        Creates a PipelineIOHandler instance from a dictionary.

        :param data: A dictionary containing the input and output specifications.
        :return: A PipelineIOHandler instance.
        """
        return cls(**data)

    def dumps(self, marshaller: Marshaller = DEFAULT_MARSHALLER) -> str:
        """
        Returns the string representation of PipelineIOHandler according to the format dictated by the `Marshaller`
        in use.

        :param marshaller: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :returns: A string representing the handler.
        """
        return marshaller.marshal(self.to_dict())

    def dump(self, fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> None:
        """
        Writes the string representation of PipelineIOHandler to the file-like object passed in the `fp` argument.

        :param fp: A file-like object ready to be written to.
        :param marshaller: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        """
        fp.write(marshaller.marshal(self.to_dict()))

    @classmethod
    def loads(cls, data: Union[str, bytes, bytearray], marshaller: Marshaller = DEFAULT_MARSHALLER) -> Self:
        """
        Creates a PipelineIOHandler object from the string representation passed in the `data` argument.

        :param data: The string representation of the handler, can be `str`, `bytes` or `bytearray`.
        :param marshaller: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :returns: A PipelineIOHandler instance.
        """
        return cls.from_dict(marshaller.unmarshal(data))

    @classmethod
    def load(cls, fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> Self:
        """
        Creates a PipelineIOHandler object from the string representation read from the file-like object passed in the
        `fp` argument.

        :param fp: A file-like object ready to be read from.
        :param marshaller: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
        :returns: A PipelineIOHandler instance.
        """
        return cls.from_dict(marshaller.unmarshal(fp.read()))

    def parse_input(self, input_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Parses the input values provided to the pipeline-level inputs and returns a dictionary with the parsed values.

        :param input_data: The input values provided to the pipeline-level inputs.
        :return: A dictionary with the parsed input values.
        """
        parsed_input: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for input_name, input_spec in self.input_specs.items():
            # If the input is provided, parse the input value
            if input_name in input_data:
                input_value = input_data[input_name]
                # Parse the input value to the connected components
                for connection in input_spec.connections:
                    receiver_component_name, receiver_socket_name = parse_connect_string(connection)
                    if not receiver_socket_name:
                        raise ValueError(f"No socket name found for connection {connection}")
                    parsed_input[receiver_component_name][receiver_socket_name] = input_value
            elif input_spec.required:
                raise ValueError(f'Input variable "{input_name}" is required but not provided.')
        return parsed_input

    def get_output_components(self) -> Set[str]:
        """
        Returns the names of the components that are connected to the pipeline-level outputs.

        :return: A set containing the names of the components connected to the pipeline-level outputs.
        """
        component_names: Set[str] = set()
        for output_name, output_spec in self.output_specs.items():
            if connection := output_spec.connection:
                component_name = parse_connect_string(connection)[0]
                component_names.add(component_name)
            else:
                raise ValueError(f'No output connection found for output variable "{output_name}": {output_spec}')
        return component_names

    def parse_output(self, raw_output_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parses the output values provided by the components connected to the pipeline-level outputs.

        :param raw_output_data: The output values provided by the components connected to the pipeline-level outputs.
        :return:n A dictionary with the parsed pipeline-level output values.
        """
        parsed_output = {}
        for output_name, output_spec in self.output_specs.items():
            if connection := output_spec.connection:
                sender_component_name, sender_socket_name = parse_connect_string(connection)
                try:
                    if not sender_socket_name:
                        raise ValueError(f"No socket name found for connection {connection}")
                    parsed_output[output_name] = raw_output_data[sender_component_name][sender_socket_name]
                except KeyError as err:
                    if output_spec.required:
                        raise ValueError(
                            f'No output value found for required output variable "{output_name}".'
                        ) from err
            else:
                raise ValueError(f'No output connection found for output variable "{output_name}": {output_spec}')
        return parsed_output

    def run_pipeline(self, pipeline: Pipeline, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the pipeline with the provided input data and returns the parsed output values.

        :param pipeline: The pipeline to run.
        :param data: The input data for the pipeline.
        :return: The parsed output values from the pipeline.
        """
        res = pipeline.run(
            self.parse_input(data),
            include_outputs_from=self.get_output_components(),
        )
        return self.parse_output(res)

    # =========================================================================
    # Type hint accessors
    # =========================================================================

    def get_input_type_hints(self) -> Dict[str, type]:
        """
        Returns a dictionary mapping input names to their types.

        :return: A dictionary of input names to types.
        """
        return {name: spec.type for name, spec in self.input_specs.items()}

    def get_output_type_hints(self) -> Dict[str, type]:
        """
        Returns a dictionary mapping output names to their types.

        :return: A dictionary of output names to types.
        """
        return {name: spec.type for name, spec in self.output_specs.items()}

    # =========================================================================
    # Request/Response serialization with type-aware deserialization
    # =========================================================================

    def deserialize_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize HTTP request data using input type hints.

        This method handles the HTTP boundary where incoming data is plain JSON
        without __type__ metadata. It uses the InputSpec types to properly
        reconstruct objects like ChatMessage.

        :param data: Raw request data (dict from JSON).
        :return: Deserialized data with proper types.
        """
        type_hints = self.get_input_type_hints()
        return {name: deserialize_value(value, type_hints.get(name)) for name, value in data.items()}

    def serialize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize response data for HTTP transmission.

        :param data: Response data with Python objects.
        :return: JSON-serializable data.
        """
        return {name: serialize_value(value) for name, value in data.items()}
