"""
This module defines MilvusVectorDatabase, a class specifically designed for managing and interacting with Milvus vector
databases.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Iterable, TYPE_CHECKING

from haystack import Document
from haystack.lazy_imports import LazyImport

from ravelin.assets._base import SecretValue, resolve_secret
from ravelin.assets.vector_databases._base import _VectorDatabase, register_vector_database

with LazyImport("Run 'pip install pymilvus'") as pymilvus_import:
    from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker, WeightedRanker
    from pymilvus.client.constants import RANKER_TYPE_RRF, RANKER_TYPE_WEIGHTED

    if TYPE_CHECKING:
        from pymilvus import SearchResult
        from pymilvus.client.abstract import BaseRanker


@register_vector_database("milvus")
class MilvusVectorDatabase(_VectorDatabase):
    """
    A VectorDatabase asset that refers to a Milvus database. Either a uri or host and port must be provided.

    :param name: The name of the vector database.
    :param collection_name: The name of the collection in the database.
    :param content_field: The field in the database that contains the raw content from which embeddings are generated.
    :param embedding_fields: A dictionary mapping field names to embedding configurations.
    :param meta_fields: The fields in the database that contain metadata.
    :param uri: The URI of the milvus server. This can be in the form of a URL or a connection string, e.g.,
        "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".
    :param host: The host of the milvus server.
    :param port: The port of the milvus server.
    :param vector_db_name: The database name in Milvus. Defaults to "default" if not provided.
    :param vector_db_user: The username for authentication with the Milvus server.
    :param vector_db_password: The password for authentication with the Milvus server as a Haystack ``Secret``.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        collection_name: str,
        content_field: str,
        embedding_fields: Dict[str, Dict[str, Any]],
        meta_fields: Optional[Iterable[str]] = None,
        uri: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        group_by: Optional[Dict[str, Any]] = None,
        vector_db_name: Optional[str] = None,
        vector_db_user: Optional[str] = None,
        vector_db_password: SecretValue = None,
    ):
        pymilvus_import.check()

        try:
            # Initialize connection to Milvus
            connections.connect(
                alias="default",
                uri=uri or "",
                host=host or "",
                port=port or "",
                user=vector_db_user or "",
                password=resolve_secret(vector_db_password) or "",
                db_name=vector_db_name or "default",
            )
        except Exception as e:
            # Extract meaningful error message from the exception
            error_msg = str(e)

            # Build connection details for debugging
            conn_details = f"uri='{uri}'" if uri else f"host='{host}', port={port}"

            # Create a clearer error message
            msg = (
                f"Failed to connect to Milvus database ({conn_details}). "
                f"Error: {error_msg}. "
                f"Please verify your connection parameters and credentials."
            )
            raise ConnectionError(msg) from e
        self.collection = Collection(collection_name)

        super().__init__(
            name=name,
            collection_name=collection_name,
            content_field=content_field,
            embedding_fields=embedding_fields,
            meta_fields=meta_fields,
            uri=uri,
            host=host,
            port=port,
            group_by=group_by,
            vector_db_name=vector_db_name,
            vector_db_user=vector_db_user,
            vector_db_password=vector_db_password,
        )

    def _normalize_results(self, result: "SearchResult") -> List[Document]:
        """
        Normalize the search result.

        :param result: A SearchResult object from Milvus search.
        :return: A list of Document objects containing the search results.
        """
        documents = []
        for hit in result[0]:
            doc = Document(
                content=hit.get(self.content_field),
                meta={key: value for key, value in hit.get("entity", {}).items() if key != self.content_field},
            )
            # Always include distance information from search. We used to include score, but it is on deprecation path.
            doc.meta["distance"] = hit.get("distance")
            documents.append(doc)
        return documents

    def _search(
        self,
        query_values: List[Any],
        anns_field: str,
        output_fields: List[str],
        top_k: int,
        timeout: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "SearchResult":
        """
        Search the database using the given query value.

        :param query_values: The query values for this field (either vector embedding or raw text)
        :param anns_field: The field in the database to search.
        :param output_fields: The fields to output in the search results.
        :param top_k: The number of results to return.
        :param timeout: The timeout for the search operation.
        :param filters: Filters to apply to the search results.
        :param kwargs: Additional keyword arguments to pass to the search method.
        :returns: The search result object from Milvus.
        :raises ValueError: If the anns_field is not found in the embedding_fields.
        """
        if anns_field not in self.embedding_fields:
            raise ValueError(f"embedding_field {anns_field} not found in embedding_fields.")
        results = self.collection.search(
            data=query_values,
            anns_field=anns_field,
            param=self.embedding_fields[anns_field].search_kwargs or {},
            limit=top_k,
            expr=filters.get("expr") if filters else None,
            partition_names=kwargs.get("partition_names"),
            output_fields=output_fields,
            timeout=timeout,
            round_decimal=kwargs.get("round_decimal", -1),
            group_by_field=self.group_by.get("field"),
        )
        return results

    def _build_search_request_by_field(
        self,
        query_values: List[Any],
        anns_field: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> "AnnSearchRequest":
        """
        Build a search request for a single field.

        :param query_values: The query value for this field (either vector embedding or raw text)
        :param anns_field: The field in the database to search.
        :param top_k: The number of results to return.
        :param filters: Filters to apply to the search results.
        :returns: A AnnSearchRequest request object which can be used to perform the search.
        :raises ValueError: If the anns_field is not found in the embedding_fields.
        """
        if anns_field not in self.embedding_fields:
            raise ValueError(f"embedding_field {anns_field} not found in embedding_fields.")
        return AnnSearchRequest(
            data=query_values,
            anns_field=anns_field,
            param=self.embedding_fields[anns_field].search_kwargs or {},
            limit=top_k,
            expr=filters.get("expr") if filters else None,
        )

    def _load_reranker(self, reranker_config: Optional[Dict[str, Any]]) -> "BaseRanker":
        """
        Load the reranker object based on the provided configuration. As per the current Milvus implementation, only
        RRFRanker and WeightedRanker are supported. If no reranker is provided, RRFRanker is used by default.

        :param reranker_config: The reranker configuration.
        :returns: The reranker object.
        :raises ValueError: If an invalid reranker strategy is provided.
        """
        if not reranker_config:
            logging.info("No reranker provided. Using RRFRanker by default for hybrid search.")
            return RRFRanker()

        strategy = reranker_config.get("strategy")
        if strategy == RANKER_TYPE_RRF:
            return RRFRanker(reranker_config.get("params", {}).get("k", 60))
        if strategy == RANKER_TYPE_WEIGHTED:
            return WeightedRanker(*reranker_config.get("params", {}).get("weights", []))
        raise ValueError(f"Invalid reranker strategy: {strategy}")

    def search_by_data(
        self,
        query_data: Dict[str, Any],
        *,
        embedding_fields: List[str],
        meta_fields: Optional[List[str]] = None,
        top_k: int,
        top_k_by_embedding_field: Optional[Union[int, Dict[str, int]]] = None,
        timeout: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        reranker: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Search the database using the given (single) query and return the top-k documents.

        :param query_data: Dictionary mapping field names to their query data. For dense vector fields,
                      this should be embedding vectors. For sparse vector fields, this should be raw query text.
        :param embedding_fields: The fields in the database to search.
        :param meta_fields: The fields to output in the search results.
        :param top_k: The number of results to return.
        :param top_k_by_embedding_field: The number of results to return for each embedding field.
        :param timeout: The timeout for the search operation.
        :param filters: Filters to apply to the search results. For MilvusVectorDatabase, this is a dictionary with a
            single key "expr" containing the filter expression. The syntax information of the expression can be found on
            https://milvus.io/docs/boolean.md.
        :param reranker: The reranker to use in hybrid search. For MilvusVectorDatabase, this is a dictionary
            representation of an RRFRanker or WeightedRanker. More information can be found on
            https://github.com/milvus-io/pymilvus/blob/master/pymilvus/client/abstract.py#L306-L339.
        :param kwargs: Additional keyword arguments to pass to the search method. Currently, only the following options
            are supported for MilvusVectorDatabase:
            `partition_names` -- A list of partition names.
            `round_decimal` -- The number of decimal places to round the search results to. -1 indicates no rounding.
            `group_by_field` -- Groups search results by a specified field to ensure diversity and avoid returning
                multiple results from the same group.
        :return: The top-k documents from the search.
        :raises ValueError: If required query data for any field is missing or if the wrong query type is provided for a
            field's vector type.
        """
        output_fields = [self.content_field] + (meta_fields or [])  # Always include content field in output

        if len(embedding_fields) == 0:
            raise ValueError("embedding_fields must not be empty.")

        if len(embedding_fields) == 1:
            # Run a single field search if only one embedding field is used
            if top_k_by_embedding_field is not None:
                logging.warning(
                    "top_k_by_embedding_field is ignored when only one embedding field is used for the search."
                )

            field = embedding_fields[0]

            # milvus search requires a list of query values to support both single and bulk searches,
            # so we wrap the query value in a list
            query_values = [query_data[field]]

            results = self._search(
                query_values=query_values,
                anns_field=field,
                top_k=top_k,
                output_fields=output_fields,
                timeout=timeout,
                filters=filters,
                **kwargs,
            )
        else:
            milvus_reranker = self._load_reranker(reranker)
            # If top_k_by_embedding_field is not provided, use top_k
            top_k_by_embedding_field = top_k_by_embedding_field or top_k
            if isinstance(top_k_by_embedding_field, int):
                top_k_by_embedding_field = {anns_field: top_k_by_embedding_field for anns_field in embedding_fields}

            requests = []

            for field in embedding_fields:
                query_values = [query_data[field]]

                requests.append(
                    self._build_search_request_by_field(
                        query_values=query_values,
                        anns_field=field,
                        top_k=top_k_by_embedding_field.get(field, top_k),
                        filters=filters,
                    )
                )

            results = self.collection.hybrid_search(
                reqs=requests,
                rerank=milvus_reranker,
                limit=top_k,
                partition_names=kwargs.get("partition_names"),
                output_fields=output_fields,
                timeout=timeout,
                round_decimal=kwargs.get("round_decimal", -1),
                group_by_field=self.group_by.get("field"),
            )

        return self._normalize_results(results)
