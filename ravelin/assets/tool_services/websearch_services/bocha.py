"""
This module contains a WebSearchService asset that refers to a deployed web search service following the Bocha WebSearch
 API format.
"""

import json
from typing import Any, Dict, List, Optional

import httpx
from haystack import Document

from ravelin.assets._base import SecretValue, resolve_secret
from ravelin.assets.tool_services._base import register_tool_service
from ravelin.assets.tool_services.websearch_services._base import _HttpxWebSearchService

BOCHA_BASE_URI = "https://api.bochaai.com/"
BOCHA_WEBSEARCH_ENDPOINT_SUFFIX = "v1/web-search"


@register_tool_service("bocha-websearch")
class BochaWebSearchService(_HttpxWebSearchService):
    """
    A WebSearchService asset that refers to a deployed web search service following the Bocha WebSearch API format.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: SecretValue = None,
        uri: str = BOCHA_BASE_URI,
        organization: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        endpoint_suffix: Optional[str] = None,
    ):
        super().__init__(name=name, api_key=api_key, uri=uri, organization=organization, client_kwargs=client_kwargs)
        if endpoint_suffix:
            self.configs["endpoint_suffix"] = endpoint_suffix
        self.search_endpoint = (
            f"{self.uri.rstrip('/')}/{(endpoint_suffix or BOCHA_WEBSEARCH_ENDPOINT_SUFFIX).lstrip('/')}"
        )

    def build_request(
        self,
        client: httpx.Client,
        query: str,
        *,
        num: Optional[int] = None,
        page: Optional[int] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> httpx.Request:

        body: Dict[str, Any] = {"query": query}
        if num:
            body["count"] = num
        if page:
            body["page"] = page
        if search_kwargs:
            body.update(search_kwargs)

        headers = {"Authorization": f"Bearer {resolve_secret(self.api_key)}", "Content-Type": "application/json"}
        request = client.build_request("POST", self.search_endpoint, headers=headers, json=body)
        return request

    def parse_response(self, response: httpx.Response) -> List[Document]:
        try:
            response_json = response.json()
        except json.JSONDecodeError as err:
            raise ValueError(
                f"The websearch service '{self.name}' returned an invalid JSON response: {response.text}"
            ) from err
        documents = []
        for idx, webpage_data in enumerate(response_json["data"]["webPages"].get("value", [])):
            content = webpage_data.get("summary") or webpage_data.get("snippet")
            document = Document(
                content=content,
                meta={
                    "position": idx,
                    "search_id": webpage_data.get("id"),
                    "title": webpage_data.get("name"),
                    "url": webpage_data.get("url"),
                    "date": webpage_data.get("dateLastCrawled"),
                    "snippet": webpage_data.get("snippet"),
                    "site_info": {"name": webpage_data.get("siteName"), "icon": webpage_data.get("siteIcon")},
                    "language": webpage_data.get("language"),
                },
            )
            documents.append(document)
        return documents
