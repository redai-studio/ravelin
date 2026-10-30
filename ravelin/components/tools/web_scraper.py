"""
This module contains the WebScraper component. The WebScraper component is used to perform web scraping using a
WebScraperService and return the parsed content as a list of Document objects.
"""

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from typing_extensions import Self

from ravelin.assets.asset_factory import ToolServiceFactory
from ravelin.assets.tool_services import WebScraperService
from ravelin.streaming import EventType, get_component_context
from ravelin.dataclasses import AssetReference


@component
class WebScraper:
    """
    Scrapes web pages using a WebScraperService and returns the parsed content as a list of Document objects.

    :param webscraper_service: An instance of the WebScraperService to use for scraping.
    :param timeout: The timeout duration for the HTTP client, defaults to 20.0.
    :param scraper_kwargs: Additional keyword arguments to pass to the web scraper service.
        The keyword arguments are specific to the web scraper service being used. Defaults to None.
    """

    def __init__(
        self,
        webscraper_service: WebScraperService,
        timeout: Optional[float] = 20.0,
        output_formats: Optional[List[str]] = None,
        scraper_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.webscraper_service = webscraper_service
        self.timeout = timeout
        self.output_formats = output_formats
        self.scraper_kwargs = scraper_kwargs or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the component.

        :return: A dictionary representation of the component.
        """
        webscraper_service_reference = self.webscraper_service.reference
        return default_to_dict(
            self,
            webscraper_service=webscraper_service_reference.to_dict(),
            timeout=self.timeout,
            output_formats=self.output_formats,
            scraper_kwargs=self.scraper_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Load a WebScraper component from its serialized dictionary representation.

        :param data: The serialized dictionary representation of the component.
        :return: The deserialized WebSearch component.
        """
        # Make a shallow copy of init_parameters to avoid modifying the original data
        init_parameters_data = data.get("init_parameters", {}).copy()
        # Load the chat model service from reference
        webscraper_service_reference = AssetReference.from_dict(init_parameters_data["webscraper_service"])
        webscraper_service = ToolServiceFactory.load_from_reference(webscraper_service_reference)
        if not isinstance(webscraper_service, WebScraperService):
            raise TypeError(
                f"The webscraper service must be an instance of WebScraperService, got {type(webscraper_service)}"
            )
        init_parameters_data["webscraper_service"] = webscraper_service
        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters_data})

    @component.output_types(documents=List[Document])
    def run(self, url: str, scraper_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run the WebScraper component to scrape the content of a web page.

        :param url: The URL of the web page to scrape.
        :param scraper_kwargs: Additional keyword arguments to pass to the web scraper service at runtime.
            These arguments will merge with the scraper_kwargs provided during initialization.
        :return:
            A dictionary with the following keys:
                - `documents`: The scraped content as a list of Document objects.
        """
        comp_ctx = get_component_context()
        if comp_ctx:
            comp_ctx.emit(EventType.COMPONENT_ACTIVITY, url, subtype="scrape_webpage")

        scraper_kwargs = {**self.scraper_kwargs, **(scraper_kwargs or {})}
        documents = self.webscraper_service(
            url,
            timeout=self.timeout,
            output_formats=self.output_formats,
            scraper_kwargs=scraper_kwargs,
        )

        if comp_ctx:
            comp_ctx.emit(EventType.COMPONENT_ACTIVITY, documents, subtype="scrape_results")
        return {"documents": documents}

    @component.output_types(documents=List[Document])
    async def run_async(self, url: str, scraper_kwargs: Optional[Dict[str, Any]] = None):
        """
        Asynchronously run the WebScraper component.

        Emits COMPONENT_ACTIVITY events before and after the scrape so that streaming
        consumers receive live progress without needing per-call callbacks.
        """
        comp_ctx = get_component_context()
        if comp_ctx:
            await comp_ctx.emit_async(EventType.COMPONENT_ACTIVITY, url, subtype="scrape_webpage")

        scraper_kwargs = {**self.scraper_kwargs, **(scraper_kwargs or {})}
        documents = self.webscraper_service(
            url,
            timeout=self.timeout,
            output_formats=self.output_formats,
            scraper_kwargs=scraper_kwargs,
        )

        if comp_ctx:
            await comp_ctx.emit_async(EventType.COMPONENT_ACTIVITY, documents, subtype="scrape_results")
        return {"documents": documents}
