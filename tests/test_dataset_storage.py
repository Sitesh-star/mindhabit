import io
import os
import tempfile
import unittest

import pandas as pd
from werkzeug.datastructures import FileStorage

import app as mindhabits
from utils.dataset_storage import (
    DatasetNotFoundError,
    InvalidDatasetError,
    create_dataset_path,
    load_dataset,
    save_local_upload,
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


if __name__ == "__main__":
    unittest.main()
