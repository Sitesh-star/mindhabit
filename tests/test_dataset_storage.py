import io
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd
from werkzeug.datastructures import FileStorage

import app as mindhabits
from utils.dataset_storage import (
    DatasetNotFoundError,
    InvalidDatasetError,
    DatasetStorageError,
    create_dataset_path,
    load_dataset,
    save_local_upload,
    validate_blob_reference,
)


class DatasetStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_unique_object_names_are_sanitized(self):
        first = create_dataset_path("../My wellness data.csv")
        second = create_dataset_path("../My wellness data.csv")
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^datasets/[0-9a-f-]{36}/My_wellness_data\.csv$")

    def test_local_reference_loads_csv(self):
        upload = FileStorage(
            stream=io.BytesIO(b"sleep,mood\n7,good\n6,low\n"),
            filename="wellness.csv",
            content_type="text/csv",
        )
        reference = save_local_upload(upload, self.directory.name)
        frame = load_dataset(reference, self.directory.name)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(frame.shape, (2, 2))

    def test_missing_and_invalid_datasets_are_reported(self):
        with self.assertRaises(DatasetNotFoundError):
            load_dataset(None, self.directory.name)
        missing = {"storage": "local", "pathname": "missing.csv", "filename": "missing.csv"}
        with self.assertRaises(DatasetNotFoundError):
            load_dataset(missing, self.directory.name)
        invalid_path = os.path.join(self.directory.name, "invalid.csv")
        with open(invalid_path, "wb") as invalid_file:
            invalid_file.write(b'"broken\n')
        invalid = {"storage": "local", "pathname": "invalid.csv", "filename": "invalid.csv"}
        with self.assertRaises(InvalidDatasetError):
            load_dataset(invalid, self.directory.name)

    def test_application_has_no_uploaded_file_path_global(self):
        self.assertFalse(hasattr(mindhabits, "uploaded_file_path"))

    def test_blob_loader_uses_exact_canonical_url(self):
        reference = {
            "storage": "blob",
            "pathname": "datasets/id/data-canonical.csv",
            "url": (
                "https://store.private.blob.vercel-storage.com/"
                "datasets/id/data-canonical.csv"
            ),
            "filename": "data.csv",
        }
        with patch(
            "utils.dataset_storage._read_blob_bytes",
            return_value=b"sleep,mood\n7,good\n",
        ) as reader:
            frame = load_dataset(reference, self.directory.name)
        reader.assert_called_once_with(reference["url"])
        self.assertEqual(frame.shape, (1, 2))

    def test_blob_reference_rejects_url_path_mismatch(self):
        reference = {
            "storage": "blob",
            "pathname": "datasets/id/expected.csv",
            "url": (
                "https://store.private.blob.vercel-storage.com/"
                "datasets/id/different.csv"
            ),
            "filename": "expected.csv",
        }
        with self.assertRaises(DatasetStorageError):
            validate_blob_reference(reference)

    def test_blob_reference_reports_404(self):
        from vercel.blob import BlobNotFoundError

        reference = {
            "storage": "blob",
            "pathname": "datasets/id/missing.csv",
            "url": (
                "https://store.private.blob.vercel-storage.com/"
                "datasets/id/missing.csv"
            ),
            "filename": "missing.csv",
        }
        with patch(
            "vercel.blob.head_blob",
            new=AsyncMock(side_effect=BlobNotFoundError("missing")),
        ):
            with self.assertRaises(DatasetNotFoundError):
                validate_blob_reference(reference)


if __name__ == "__main__":
    unittest.main()
