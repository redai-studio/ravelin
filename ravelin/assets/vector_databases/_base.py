"""
This module defines the base class for vector databases. A vector database is a database that stores vector
representations of data. Vector databases are widely used as the backend for retrieval-based applications that require
fast and efficient similarity search to find the most relevant data points, playing a crucial role in Retrieval
Augmented Generation (RAG) applications.
"""

import abc
from typing import Any, Dict, Iterable, List, Optional, Union, Type, Callable

from haystack import Document

from ravelin.assets._base import _Asset, SecretValue
from ravelin.assets.resolvers import normalize_asset_secrets
from ravelin.dataclasses import EmbeddingField
from ravelin.utils.registry import RegistryMeta, Registry

# Create a registry for vector database types
VECTOR_DATABASE_REGISTRY = Registry["_VectorDatabase"]("VECTOR_DB_TYPE")

# Create decorator for registering vector databases
register_vector_database: Callable[[str], Callable[[Type["_VectorDatabase"]], Type["_VectorDatabase"]]] = (
    VECTOR_DATABASE_REGISTRY.create_decorator()
)


class _VectorDatabase(_Asset, metaclass=RegistryMeta):
    """
    A VectorDatabase asset refers to a database that stores vector representations of data. Either a uri or host
    and port must be provided to connect to the database.

    :param name: The name of the vector database.
    :param collection_name: The name of the collection in the database.
    :param content_field: The field in the database that contains the raw content from which embeddings are generated.
    :param embedding_fields: A dictionary mapping field names to embedding configurations.
    :param meta_fields: The fields in the database that contain metadata.
    :param uri: The URI of the vector database server. This can be in the form of a URL or a connection string, e.g.,
        "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".
    :param host: The host of the vector database server.
    :param port: The port of the vector database server.
    :param vector_db_name: The database name in the vector database system.
    :param vector_db_user: The username for authentication with the vector database server.
    :param vector_db_password: The password for authentication with the vector database server as a Haystack
        ``Secret``.
    """

    ASSET_TYPE = "vector-database"
    TYPE_IDENTIFIER = "VECTOR_DB_TYPE"

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
        if not uri and not (host and port):
            raise ValueError("Either a uri or host and port must be provided to connect to the vector database.")
        configs: Dict[str, Any] = {
            "vector_db_type": self.vector_db_type,
            "collection_name": collection_name,
            "content_field": content_field,
            "embedding_fields": embedding_fields,
            "meta_fields": list(meta_fields) if meta_fields else [],
            "uri": uri,
            "host": host,
            "port": port,
            "group_by": group_by,
            "vector_db_name": vector_db_name,
            "vector_db_user": vector_db_user,
        }
        secrets: Dict[str, Any] = {
            "vector_db_password": vector_db_password,
        }

        self.embedding_fields = {key: EmbeddingField.from_dict(value) for key, value in embedding_fields.items()}
        self.content_field = content_field
        self.meta_fields = set(meta_fields if meta_fields else ())
        self.group_by = group_by or {}
        super().__init__(name, configs=configs, secrets=secrets)

    @property
    def vector_db_type(self) -> str:
        """Get the vector db type."""
        return getattr(self, f"_{self.__class__.__name__}__VECTOR_DB_TYPE")

    @property
    def vector_db_password(self) -> SecretValue:
        """Get the password secret; resolve it with ``resolve_secret`` before use."""
        return self.secrets.get("vector_db_password")

    @vector_db_password.setter
    def vector_db_password(self, value: SecretValue) -> None:
        """Set the vector database password."""
        self.secrets["vector_db_password"] = normalize_asset_secrets({"secrets": {"vector_db_password": value}})[
            "secrets"
        ]["vector_db_password"]

    def validate_meta_fields(self, meta_fields: Iterable[str]) -> None:
        """
        Validate that the meta fields are a subset of the available meta fields in the database.

        :param meta_fields: The meta fields to validate.
        :raises ValueError: If the meta fields are not a subset of the available meta fields in the database.
        """
        if not set(meta_fields).issubset(self.meta_fields):
            raise ValueError(
                f"meta_fields must be a subset of the available meta fields ({self.meta_fields}) in the database, "
                f"got: {meta_fields}"
            )

    def _embed_query(self, raw_query: Any, embedding_fields: List[str]) -> Dict[str, Any]:
        """
        Embed a raw query for the given embedding fields.

        :param raw_query: The raw query to embed.
        :param embedding_fields: The fields to embed.
        :return: A dictionary mapping embedding field names to their embeddings.
        """
        query_data = {}
        for embedding_field_name in embedding_fields:
            if embedding_field_name not in self.embedding_fields:
                raise ValueError(f"Embedding field {embedding_field_name} not found in embedding_fields")

            embedding_field = self.embedding_fields[embedding_field_name]
            if embedding_field.embedding_function is not None:
                query_data[embedding_field_name] = embedding_field.embedding_function([raw_query])[0]
            else:
                query_data[embedding_field_name] = raw_query
        return query_data

    @abc.abstractmethod
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
        Search the database using the given query and return the top-k documents.

        :param query_data: Dictionary mapping field names to their query data. For dense vector fields,
                        this should be embedding vectors. For sparse vector fields, this should be raw query text.
        :param embedding_fields: The fields in the database to search.
        :param meta_fields: The fields to output in the search results. If None, all meta fields will be returned.
        :param top_k: The number of results to return.
        :param top_k_by_embedding_field: The number of results to return for each embedding field.
        :param timeout: The timeout for the search operation.
        :param filters: Filters to apply to the search results.
        :param reranker: The reranker to use in hybrid search.
        :param kwargs: Additional keyword arguments to pass to the search method.
        :return: The top-k documents from the search.
        :raises ValueError: If required query data for any field is missing or if the wrong query type is provided for a
            field's vector type.
        """
        raise NotImplementedError("search method must be implemented")

    def search(
        self,
        query: Any,
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
        Search the database using the given query and return the top-k documents.

        :param query_data: Dictionary mapping field names to their query data. For dense vector fields,
                        this should be embedding vectors. For sparse vector fields, this should be raw query text.
        :param embedding_fields: The fields in the database to search.
        :param meta_fields: The fields to output in the search results. If None, all meta fields will be returned.
        :param top_k: The number of results to return.
        :param top_k_by_embedding_field: The number of results to return for each embedding field.
        :param timeout: The timeout for the search operation.
        :param filters: Filters to apply to the search results.
        :param reranker: The reranker to use in hybrid search.
        :param kwargs: Additional keyword arguments to pass to the search method.
        :return: The top-k documents from the search.
        :raises ValueError: If required query data for any field is missing or if the wrong query type is provided for a
            field's vector type.
        """
        query_data = self._embed_query(query, embedding_fields)
        return self.search_by_data(
            query_data,
            embedding_fields=embedding_fields,
            meta_fields=meta_fields,
            top_k=top_k,
            top_k_by_embedding_field=top_k_by_embedding_field,
            timeout=timeout,
            filters=filters,
            reranker=reranker,
            **kwargs,
        )
