"""Dataset storage shared by upload, preview, analysis, and export routes."""

import asyncio
from io import BytesIO
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from werkzeug.utils import secure_filename


MAX_DATASET_BYTES = 25 * 1024 * 1024


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


def make_blob_reference(pathname, filename):
    return {
        "storage": "blob",
        "pathname": pathname,
        "filename": secure_filename(filename or "dataset.csv") or "dataset.csv",
    }


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
        content = _read_blob_bytes(reference["pathname"])
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
