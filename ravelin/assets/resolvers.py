"""Asset resolver registration and secret normalization."""

from typing import Any, Dict, Mapping, Protocol

from haystack.utils import Secret
from haystack.utils.auth import TokenSecret

from ravelin.dataclasses.asset_reference import AssetReference
from ravelin.utils.asset_storage import get_local_asset

LOCAL_ASSET_SOURCE = "local"


class AssetResolver(Protocol):
    """Resolve an asset reference into data understood by an asset factory."""

    def resolve(self, reference: AssetReference) -> Mapping[str, Any]:
        """Resolve an asset reference."""
        raise NotImplementedError  # pragma: no cover


def normalize_asset_secrets(
    data: Mapping[str, Any],
    *,
    allow_plaintext: bool = False,
    allow_token_secrets: bool = True,
) -> Dict[str, Any]:
    """Copy asset data and normalize its secret values into Haystack ``Secret`` objects.

    Plain strings can only be accepted at an explicitly trusted resolver boundary. They are converted into
    non-serializable token secrets. Serialized environment-variable secrets are converted back into ``Secret``
    objects. Error messages deliberately identify only the field, never its value.
    """
    normalized = dict(data)
    raw_secrets = normalized.get("secrets")
    if raw_secrets is None:
        raw_secrets = {}
    if not isinstance(raw_secrets, Mapping):
        raise TypeError("Asset 'secrets' must be a mapping.")

    secrets: Dict[str, Secret | None] = {}
    for name, value in raw_secrets.items():
        if not isinstance(name, str):
            raise TypeError("Asset secret names must be strings.")
        if value is None:
            secrets[name] = None
            continue
        if isinstance(value, Secret):
            secret = value
        elif isinstance(value, Mapping):
            secret = Secret.from_dict(dict(value))
        elif isinstance(value, str) and allow_plaintext:
            secret = Secret.from_token(value)
        else:
            raise TypeError(
                f"Asset secret '{name}' must be a Haystack Secret or a serialized environment-variable reference."
            )
        if isinstance(secret, TokenSecret) and not allow_token_secrets:
            raise ValueError(f"Local asset secret '{name}' must use an environment-variable reference.")
        secrets[name] = secret

    normalized["secrets"] = secrets
    return normalized


class LocalAssetResolver:
    """Resolve local asset JSON without permitting embedded secret values."""

    def resolve(self, reference: AssetReference) -> Mapping[str, Any]:
        """Read and normalize a local asset definition."""
        return normalize_asset_secrets(
            get_local_asset(reference.name),
            allow_token_secrets=False,
        )


_ASSET_RESOLVERS: Dict[str, AssetResolver] = {LOCAL_ASSET_SOURCE: LocalAssetResolver()}


def register_asset_resolver(source: str, resolver: AssetResolver) -> None:
    """Register a resolver for a non-local asset source."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Asset resolver source must be a non-empty string.")
    if source == LOCAL_ASSET_SOURCE:
        raise ValueError(f"Asset resolver source '{LOCAL_ASSET_SOURCE}' is reserved by Ravelin.")
    if source in _ASSET_RESOLVERS:
        raise ValueError(f"An asset resolver is already registered for source '{source}'.")
    if not callable(getattr(resolver, "resolve", None)):
        raise TypeError("Asset resolver must provide a callable resolve(reference) method.")
    _ASSET_RESOLVERS[source] = resolver


def resolve_asset(reference: AssetReference) -> Dict[str, Any]:
    """Resolve an asset reference with its explicitly registered resolver."""
    try:
        resolver = _ASSET_RESOLVERS[reference.source]
    except KeyError as err:
        raise ValueError(
            f"No asset resolver is registered for source '{reference.source}'. "
            "Register one with register_asset_resolver()."
        ) from err
    return dict(resolver.resolve(reference))
