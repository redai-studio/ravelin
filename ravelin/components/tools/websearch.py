"""
This module contains the WebSearch component. The WebSearch component is used to perform a web search using a
WebSearchService and return the search results as a list of Document objects.
"""

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from typing_extensions import Self

from ravelin.assets.asset_factory import ToolServiceFactory
from ravelin.assets.tool_services import WebSearchService
from ravelin.streaming import EventType, get_component_context
from ravelin.dataclasses import AssetReference


@component
class WebSearch:
    """
    Performs a web search using a WebSearchService and returns the search results as a list of Document objects.

    :param websearch_service: An instance of the WebSearchService to use for the web search.
    :param num: The number of search results to return, defaults to None (i.e., the service default).
    :param page: The page number of search results to return, defaults to None (i.e., the service default).
    :param timeout: The timeout duration for the HTTP client, defaults to 20.0.
    :param search_kwargs: Additional keyword arguments to pass to the web search service.
        The keyword arguments are specific to the web search service being used. Defaults to None.
    """

    def __init__(
        self,
        websearch_service: WebSearchService,
        num: Optional[int] = None,
        page: Optional[int] = None,
        timeout: Optional[float] = 20.0,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.websearch_service = websearch_service
        self.num = num
        self.page = page
        self.timeout = timeout
        self.search_kwargs = search_kwargs or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the component.

        :return: A dictionary representation of the component.
        """
        websearch_service_reference = self.websearch_service.reference
        return default_to_dict(
            self,
            websearch_service=websearch_service_reference.to_dict(),
            num=self.num,
            page=self.page,
            timeout=self.timeout,
            search_kwargs=self.search_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Load a WebSearch component from its serialized dictionary representation.

        :param data: The serialized dictionary representation of the component.
        :return: The deserialized WebSearch component.
        """
        # Make a shallow copy of init_parameters to avoid modifying the original data
        init_parameters_data = data.get("init_parameters", {}).copy()
        # Load the chat model service from reference
        websearch_service_reference = AssetReference.from_dict(init_parameters_data["websearch_service"])
        websearch_service = ToolServiceFactory.load_from_reference(websearch_service_reference)
        if not isinstance(websearch_service, WebSearchService):
            raise TypeError(
                f"The websearch service must be an instance of WebSearchService, got {type(websearch_service)}"
            )
        init_parameters_data["websearch_service"] = websearch_service
        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters_data})

    @component.output_types(documents=List[Document])
    def run(self, query: str, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run the WebSearch component to perform a web search using the configured WebSearchService.

        :param query: The search query.
        :param search_kwargs: Additional keyword arguments to pass to the web search service at runtime.
            These arguments will merge with the search_kwargs provided during initialization.
        :return:
            A dictionary with the following keys:
                - `documents`: The search results as a list of Document objects.
        """
        comp_ctx = get_component_context()
        if comp_ctx:
            comp_ctx.emit(EventType.COMPONENT_ACTIVITY, query, subtype="search_query")

        search_kwargs = {**self.search_kwargs, **(search_kwargs or {})}
        documents = self.websearch_service(
            query,
            num=self.num,
            page=self.page,
            timeout=self.timeout,
            search_kwargs=search_kwargs,
        )

        if comp_ctx:
            comp_ctx.emit(EventType.COMPONENT_ACTIVITY, documents, subtype="search_results")
        return {"documents": documents}

    @component.output_types(documents=List[Document])
    async def run_async(self, query: str, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Asynchronously run the WebSearch component.

        Emits COMPONENT_ACTIVITY events before and after the search so that streaming
        consumers receive live progress without needing per-call callbacks.
        """
        comp_ctx = get_component_context()
        if comp_ctx:
            await comp_ctx.emit_async(EventType.COMPONENT_ACTIVITY, query, subtype="search_query")

        search_kwargs = {**self.search_kwargs, **(search_kwargs or {})}
        documents = self.websearch_service(
            query,
            num=self.num,
            page=self.page,
            timeout=self.timeout,
            search_kwargs=search_kwargs,
        )

        if comp_ctx:
            await comp_ctx.emit_async(EventType.COMPONENT_ACTIVITY, documents, subtype="search_results")
        return {"documents": documents}
