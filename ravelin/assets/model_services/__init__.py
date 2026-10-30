"""
This module provides a collection of classes that represent and manage model services in the application.

A model service is a deployed model (or a selection of models) that can be used for inference. Model services can be
used to generate responses to input data, such as text or images, and have wide-ranging use cases in an application.
"""

from ravelin.assets.model_services._base import (
    _HttpModelService as HttpModelService,
    _ModelService as ModelService,
    MODEL_SERVICE_REGISTRY,
    register_model_service,
)
from ravelin.assets.model_services.chat_model_services import *  # noqa: F403
from ravelin.assets.model_services.embedding_model_services import *  # noqa: F403

__all__ = [
    "ModelService",
    "HttpModelService",
    "MODEL_SERVICE_REGISTRY",
    "register_model_service",
]
