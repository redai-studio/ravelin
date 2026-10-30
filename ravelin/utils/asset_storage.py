"""
This module provides utility functions to manage asset storage.
"""

import json
import os
from pathlib import Path
from typing import Union, Any, Dict

ENV_LOCAL_ASSET_STORAGE_PATH = "LOCAL_ASSET_STORAGE_PATH"


def set_local_asset_storage_path(path: Union[Path, str]) -> None:
    """
    Set the local asset storage path. This path is used to store assets locally.

    :param path: The local asset storage path.
    """
    os.environ[ENV_LOCAL_ASSET_STORAGE_PATH] = str(path)


def get_local_asset_storage_path() -> Path:
    """
    Get the local asset storage path.

    :return: The local asset storage path.
    """
    try:
        return Path(os.environ[ENV_LOCAL_ASSET_STORAGE_PATH])
    except KeyError as err:
        raise ValueError(
            f"Environment variable {ENV_LOCAL_ASSET_STORAGE_PATH} is not set. "
            "Please set it to the local asset storage path via set_local_asset_storage_path()."
        ) from err


def get_local_asset_path(asset_name: str) -> Path:
    """
    Resolve an asset file beneath the configured local storage directory.

    :param asset_name: A relative name with optional forward-slash-separated
        subdirectories. Empty, dot and parent segments, backslashes, colons and
        NUL characters are not permitted.
    :return: The resolved absolute path of the asset JSON file.
    :raises ValueError: If the name is invalid or a symlink escapes the directory.
    """
    if not isinstance(asset_name, str) or not asset_name.strip():
        raise ValueError("Local asset name must be a non-empty string.")
    if any(char in asset_name for char in ("\0", "\\", ":")) or any(
        part in ("", ".", "..") for part in asset_name.split("/")
    ):
        raise ValueError(
            "Local asset name must contain relative forward-slash-separated names "
            "without empty, dot or parent segments, backslashes, colons or NUL characters."
        )

    root = get_local_asset_storage_path().resolve()
    path = (root / f"{asset_name}.json").resolve()
    if root not in path.parents:
        raise ValueError("Local asset path must remain within the configured storage directory.")
    return path


def get_local_asset(asset_name: str) -> Dict[str, Any]:
    """
    Get the metadata of the asset file for the given asset name. This function
    also checks if the asset file exists, and raises FileNotFoundError if the asset file is not found.

    :param asset_name: The name of the asset.
    :return: The metadata of the asset file.
    :raises FileNotFoundError: If the asset file is not found.
    """
    asset_path = get_local_asset_path(asset_name)
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset file {asset_path} not found.")
    with open(asset_path, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    return data
