import json
import unittest

import httpx
from haystack import Document
from haystack.utils import Secret

from ravelin.assets.tool_services.websearch_services.bocha import BochaWebSearchService


class TestBochaWebSearchService(unittest.TestCase):
    def setUp(self):
        self.websearch_service = BochaWebSearchService(
            name="test_bocha_websearch_service",
            api_key=Secret.from_token("test_key"),
            organization="test_org",
        )
        self.client = httpx.Client()

    def test_initialization(self):
        self.assertEqual("test_bocha_websearch_service", self.websearch_service.name)
        self.assertEqual(
            {
                "uri": "https://api.bochaai.com/",
                "organization": "test_org",
                "call_api_format": "bocha-websearch",
            },
            self.websearch_service.configs,
        )
        self.assertEqual("test_key", self.websearch_service.secrets["api_key"].resolve_value())
        self.assertEqual("https://api.bochaai.com/v1/web-search", self.websearch_service.search_endpoint)

    def test_custom_endpoint(self):
        websearch_service = BochaWebSearchService(
            uri="https://www.example.com/",
            endpoint_suffix="/foobar",
            api_key=Secret.from_token("test_key"),
        )
        self.assertEqual(
            {
                "uri": "https://www.example.com/",
                "call_api_format": "bocha-websearch",
                "endpoint_suffix": "/foobar",
                "organization": None,
            },
            websearch_service.configs,
        )
        self.assertEqual("test_key", websearch_service.secrets["api_key"].resolve_value())
        self.assertEqual("https://www.example.com/foobar", websearch_service.search_endpoint)

    def test_build_request(self):
        request = self.websearch_service.build_request(
            self.client, "test_query", num=10, page=1, search_kwargs={"summary": True}
        )
        self.assertEqual("POST", request.method)
        self.assertEqual("https://api.bochaai.com/v1/web-search", request.url)
        self.assertDictEqual(
            {
                "query": "test_query",
                "count": 10,
                "page": 1,
                "summary": True,
            },
            json.loads(request.content),
        )

    def test_parse_response(self):
        response_body = {
            "code": 200,
            "log_id": "d71841ad20095f61",
            "msg": None,
            "data": {
                "_type": "SearchResponse",
                "queryContext": {"originalQuery": "阿里巴巴2024年的esg报告"},
                "webPages": {
                    "webSearchUrl": "",
                    "totalEstimatedMatches": 8912791,
                    "value": [
                        {
                            "id": None,
                            "name": "阿里巴巴发布2024年ESG报告　持续推进减碳与数字化普惠",
                            "url": "https://www.alibabagroup.com/document-1752073403914780672",
                            "displayUrl": "https://www.alibabagroup.com/document-1752073403914780672",
                            "snippet": "阿里巴巴集团发布《2024财年环境、社会和治理（ESG）报告》...",
                            "summary": (
                                "阿里巴巴集团发布《2024财年环境、社会和治理（ESG）报告》（下称“报告”），"
                                "详细分享过去一年在ESG各方面取得的进展。"
                            ),
                            "siteName": "www.alibabagroup.com",
                            "siteIcon": (
                                "https://th.bochaai.com/favicon?domain_url="
                                "https://www.alibabagroup.com/document-1752073403914780672"
                            ),
                            "dateLastCrawled": "2024-07-22T00:00:00Z",
                            "cachedPageUrl": None,
                            "language": None,
                            "isFamilyFriendly": None,
                            "isNavigational": None,
                        },
                        {
                            "id": None,
                            "name": "阿里巴巴发布2024年ESG报告: 保持前瞻、保持善意、保持务实",
                            "url": "https://mparticle.uc.cn/article_org.html",
                            "displayUrl": "https://mparticle.uc.cn/article_org.html",
                            "snippet": "2024财年是阿里巴巴深化ESG治理的一年...",
                            "siteName": "大鱼号",
                            "siteIcon": (
                                "https://th.bochaai.com/favicon?domain_url=" "https://mparticle.uc.cn/article_org.html"
                            ),
                            "dateLastCrawled": "2024-07-22T11:54:00Z",
                            "cachedPageUrl": None,
                            "language": None,
                            "isFamilyFriendly": None,
                            "isNavigational": None,
                        },
                    ],
                    "someResultsRemoved": True,
                },
                "images": {},
                "videos": None,
            },
        }
        response = httpx.Response(
            status_code=200,
            content=json.dumps(response_body).encode(),
        )
        documents = self.websearch_service.parse_response(response)

        self.assertEqual(2, len(documents))
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))
        for idx, doc in enumerate(documents):
            self.assertEqual(
                response_body["data"]["webPages"]["value"][idx].get("summary")
                or response_body["data"]["webPages"]["value"][idx]["snippet"],
                doc.content,
            )
            self.assertEqual(
                {
                    "position": idx,
                    "search_id": None,
                    "title": response_body["data"]["webPages"]["value"][idx]["name"],
                    "url": response_body["data"]["webPages"]["value"][idx]["url"],
                    "date": response_body["data"]["webPages"]["value"][idx]["dateLastCrawled"],
                    "snippet": response_body["data"]["webPages"]["value"][idx]["snippet"],
                    "site_info": {
                        "name": response_body["data"]["webPages"]["value"][idx]["siteName"],
                        "icon": response_body["data"]["webPages"]["value"][idx]["siteIcon"],
                    },
                    "language": response_body["data"]["webPages"]["value"][idx]["language"],
                },
                doc.meta,
            )


if __name__ == "__main__":
    unittest.main()
