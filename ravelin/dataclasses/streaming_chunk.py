"""
This file contains the dataclass for the StreamingChunk object. This object is used to represent a chunk of streaming
data from a model service. It is inherited from the StreamingChunk object in the Haystack library, with an additional
field for reasoning content.
"""

from dataclasses import dataclass
from typing import Optional

from haystack.dataclasses import StreamingChunk as _StreamingChunk


@dataclass
class StreamingChunk(_StreamingChunk):
    """
    Data class representing a chunk of streaming data from a model service.

    This class extends Haystack's StreamingChunk with an additional field for reasoning content.
    The tool_calls field is now provided by the parent class.
    """

    reasoning_content: Optional[str] = None
