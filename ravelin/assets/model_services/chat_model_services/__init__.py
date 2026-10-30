"""
This module provides a collection of classes that represent and manage chat model services in the application.

A chat model service is a deployed model (or a selection of models) that can be used for generating replies to chat
message inputs.

Chat model services include:

- ChatModelService:
    An abstract base class of a generic service for managing and interacting with chat models.
- HttpxChatModelService:
    An abstract base class of a generic service for managing and interacting with chat models using HTTPX.
- OpenAIChatModelService:
    A service specifically designed for managing and interacting with OpenAI-API style chat models.
- OpenAIResponseModelService:
    A service for the OpenAI Responses API, covering both OpenAI and Azure OpenAI deployments.
"""

from ravelin.assets.model_services.chat_model_services._base import (
    _ChatModelService as ChatModelService,
    _HttpxChatModelService as HttpxChatModelService,
)
from ravelin.assets.model_services.chat_model_services.amazon_bedrock import BedrockAnthropicChatModelService
from ravelin.assets.model_services.chat_model_services.anthropic import AnthropicChatModelService
from ravelin.assets.model_services.chat_model_services.google import GeminiChatModelService
from ravelin.assets.model_services.chat_model_services.openai import OpenAIChatModelService
from ravelin.assets.model_services.chat_model_services.openai_response import OpenAIResponseModelService

__all__ = [
    # Abstract base classes
    "ChatModelService",
    "HttpxChatModelService",
    # Supported chat model services
    "AnthropicChatModelService",
    "BedrockAnthropicChatModelService",
    "GeminiChatModelService",
    "OpenAIChatModelService",
    "OpenAIResponseModelService",
]
