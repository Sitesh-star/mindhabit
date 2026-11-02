"""Dataset storage shared by upload, preview, analysis, and export routes."""

import asyncio
from io import BytesIO
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
from werkzeug.utils import secure_filename


MAX_DATASET_BYTES = 25 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


class DatasetStorageError(Exception):
    """A dataset could not be stored or loaded safely."""


class DatasetNotFoundError(DatasetStorageError):
    """The selected dataset no longer exists."""


class InvalidDatasetError(DatasetStorageError):
    """The selected object is not a readable CSV dataset."""


def blob_storage_enabled():
    return bool(os.environ.get("BLOB_READ_WRITE_TOKEN") or os.environ.get("BLOB_STORE_ID"))


def create_dataset_path(filename):
    safe_name = secure_filename(filename or "dataset.csv") or "dataset.csv"
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"
    return f"datasets/{uuid4()}/{safe_name}"


def make_blob_reference(pathname, filename, url=None):
    reference = {
        "storage": "blob",
        "pathname": pathname,
        "filename": secure_filename(filename or "dataset.csv") or "dataset.csv",
    }
    if url:
        reference["url"] = url
    return reference


def validate_blob_reference(reference):
    """Verify canonical private Blob metadata without downloading the CSV."""
    if not isinstance(reference, dict) or reference.get("storage") != "blob":
        raise DatasetStorageError("The uploaded dataset is not a Blob reference.")
    pathname = reference.get("pathname")
    url = reference.get("url")
    filename = reference.get("filename")
    if not isinstance(pathname, str) or not pathname.startswith("datasets/"):
        raise DatasetStorageError("The uploaded Blob pathname is invalid.")
    if ".." in pathname.split("/") or pathname.endswith("/"):
        raise DatasetStorageError("The uploaded Blob pathname is invalid.")
    if (
        not isinstance(filename, str)
        or not filename.lower().endswith(".csv")
        or secure_filename(filename) != filename
    ):
        raise DatasetStorageError("The uploaded dataset filename is invalid.")
    if url is not None and not isinstance(url, str):
        raise DatasetStorageError("The uploaded Blob reference is incomplete.")
    target = pathname
    if url:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".private.blob.vercel-storage.com")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise DatasetStorageError("The uploaded Blob reference is not private storage.")
        target = url
        LOGGER.info("Canonical uploaded Blob URL host: %s", parsed.hostname)
    try:
        from vercel.blob import BlobNotFoundError, head_blob
        metadata = asyncio.run(head_blob(target))
    except BlobNotFoundError as error:
        raise DatasetNotFoundError("The uploaded dataset could not be found.") from error
    except Exception as error:
        raise DatasetStorageError("The uploaded dataset could not be verified.") from error
    LOGGER.info("Verified canonical uploaded Blob pathname: %s", pathname)
    return reference


def save_local_upload(file_storage, upload_folder):
    """Development fallback used only when a Blob store is not configured."""
    filename = secure_filename(file_storage.filename or "")
    if not filename or not filename.lower().endswith(".csv"):
        raise InvalidDatasetError("Please choose a valid CSV file.")

    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size <= 0 or size > MAX_DATASET_BYTES:
        raise InvalidDatasetError("The CSV must be non-empty and no larger than 25 MB.")

    relative_path = create_dataset_path(filename)
    destination = Path(upload_folder) / Path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(destination)
    return {
        "storage": "local",
        "pathname": relative_path,
        "filename": filename,
    }


def dataset_filename(reference):
    return (reference or {}).get("filename") or "dataset.csv"


def dataset_identity(reference):
    if not isinstance(reference, dict):
        return None
    storage = reference.get("storage")
    pathname = reference.get("pathname")
    return f"{storage}:{pathname}" if storage and pathname else None


def _read_blob_bytes(pathname):
    try:
        from vercel.blob import BlobNotFoundError, get_blob
    except ImportError as error:
        raise DatasetStorageError("Vercel Blob support is not installed.") from error

    try:
        parsed = urlparse(pathname)
        if parsed.scheme == "https":
            LOGGER.info("Loading Blob dataset using canonical URL host: %s", parsed.hostname)
        else:
            LOGGER.info("Loading Blob dataset pathname: %s", pathname)
        result = asyncio.run(get_blob(pathname, access="private"))
        return result.body
    except BlobNotFoundError as error:
        raise DatasetNotFoundError("The dataset could not be found.") from error
    except Exception as error:
        raise DatasetStorageError("The dataset could not be downloaded.") from error


def _read_local_bytes(reference, upload_folder):
    root = Path(upload_folder).resolve()
    candidate = (root / reference["pathname"]).resolve()
    if root not in candidate.parents:
        raise DatasetStorageError("Invalid local dataset reference.")
    try:
        return candidate.read_bytes()
    except FileNotFoundError as error:
        raise DatasetNotFoundError("The dataset could not be found.") from error
    except OSError as error:
        raise DatasetStorageError("The dataset could not be read.") from error


def load_dataset(reference, upload_folder="uploads"):
    if not isinstance(reference, dict) or not reference.get("pathname"):
        raise DatasetNotFoundError("No dataset is currently selected.")

    storage = reference.get("storage")
    if storage == "blob":
        content = _read_blob_bytes(reference.get("url") or reference["pathname"])
    elif storage == "local":
        content = _read_local_bytes(reference, upload_folder)
    else:
        raise DatasetStorageError("The dataset reference is invalid.")

    if not content or len(content) > MAX_DATASET_BYTES:
        raise InvalidDatasetError("The CSV is empty or exceeds the 25 MB limit.")

    try:
        dataframe = pd.read_csv(BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise InvalidDatasetError("The file is not a readable CSV dataset.") from error
    if dataframe.empty and not len(dataframe.columns):
        raise InvalidDatasetError("The CSV dataset is empty.")
    return dataframe
