import os
import io
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import app as mindhabits


class UploadPageTests(unittest.TestCase):
    def setUp(self):
        self.previous_testing = mindhabits.app.config["TESTING"]
        mindhabits.app.config["TESTING"] = True
        self.client = mindhabits.app.test_client()

    def tearDown(self):
        mindhabits.app.config["TESTING"] = self.previous_testing

    def test_upload_page_has_one_drop_zone_and_one_submit_button(self):
        response = self.client.get("/upload")
        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(page.count('id="csvFile"'), 1)
        self.assertEqual(page.count('id="dropZone"'), 1)
        self.assertEqual(page.count('id="continueButton"'), 1)
        self.assertEqual(page.count("Validate &amp; Continue"), 1)
        self.assertIn("Drop CSV file here", page)
        self.assertIn("or click to browse", page)
        self.assertIn("No file selected", page)
        self.assertIn("Accepted format: CSV", page)
        self.assertIn("One dataset at a time", page)
        self.assertIn("Flexible Columns", page)
        self.assertIn("Automatic Validation", page)
        self.assertIn(
            "Your uploaded data is used only within MindHabits for analysis.", page
        )
        self.assertNotIn("Choose a CSV file", page)
        self.assertNotIn("Required Columns", page)
        self.assertNotIn("File Validation", page)
        self.assertIn("Enter Data Manually", page)

    def test_upload_javascript_is_valid(self):
        script_path = Path(mindhabits.app.root_path) / "static" / "js" / "static.js"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("Selected file:", script)
        self.assertIn("DataTransfer", script)
        self.assertIn('continueButton.disabled = !valid', script)
        self.assertIn('xhr.send(file)', script)
        self.assertIn('signed.uploadUrl', script)
        self.assertNotIn('new FormData', script)
        self.assertIn('storage: "blob"', script)

        node = shutil.which("node")
        if node is not None:
            result = subprocess.run(
                [node, "--check", str(script_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_production_form_cannot_submit_file_bytes_to_flask(self):
        with patch.dict(os.environ, {"VERCEL": "1", "BLOB_READ_WRITE_TOKEN": "test"}):
            page = self.client.get("/upload").get_data(as_text=True)

        form = page.split('id="uploadForm"', 1)[1].split("</form>", 1)[0]
        file_input = form.split('id="csvFile"', 1)[1].split(">", 1)[0]
        self.assertIn('data-blob-upload="true"', form)
        self.assertNotIn('multipart/form-data', form)
        self.assertNotIn('method="POST"', form)
        self.assertNotIn('action="/upload"', form)
        self.assertNotIn('name="file"', file_input)
        self.assertIn('type="button"', form)

    def test_blob_completion_sends_only_validated_metadata(self):
        environment = {"VERCEL": "1", "BLOB_READ_WRITE_TOKEN": "test"}
        with patch.dict(os.environ, environment):
            prepared = self.client.post("/upload/prepare", json={
                "filename": "wellness data.csv",
                "size": 18 * 1024 * 1024,
                "content_type": "text/csv",
            })
            self.assertEqual(prepared.status_code, 200)
            metadata = prepared.get_json()
            self.assertNotIn("file", metadata)
            self.assertNotIn("data", metadata)
            canonical_pathname = metadata["pathname"].replace(
                ".csv", "-canonical.csv"
            )
            canonical_url = (
                "https://store.private.blob.vercel-storage.com/"
                f"{canonical_pathname}"
            )
            with (
                patch("app.validate_blob_reference", side_effect=lambda value: value),
                patch("app.load_dataset", return_value=mindhabits.pd.DataFrame({"sleep": [7]})),
            ):
                completed = self.client.post("/upload/complete", json={
                    "storage": "blob",
                    "requested_pathname": metadata["pathname"],
                    "pathname": canonical_pathname,
                    "url": canonical_url,
                    "filename": metadata["filename"],
                })

        self.assertEqual(completed.status_code, 200)
        with self.client.session_transaction() as browser_session:
            reference = browser_session["dataset_reference"]
        self.assertEqual(reference["storage"], "blob")
        self.assertEqual(reference["pathname"], canonical_pathname)
        self.assertEqual(reference["url"], canonical_url)
        self.assertEqual(reference["filename"], "wellness_data.csv")
        self.assertEqual(
            browser_session["active_dataset"]["reference"], reference
        )

    def test_local_upload_validation_messages_are_specific(self):
        cases = [
            ("data.txt", b"a,b\n1,2\n", b"Invalid file format"),
            ("empty.csv", b"", b"selected CSV file is empty"),
            ("broken.csv", b"\xff\xfe\x00\x00", b"could not read this CSV file"),
        ]
        with patch("app._direct_blob_upload_enabled", return_value=False):
            for filename, content, message in cases:
                with self.subTest(filename=filename):
                    response = self.client.post(
                        "/upload",
                        data={"file": (io.BytesIO(content), filename)},
                        content_type="multipart/form-data",
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(message, response.data)

            oversized = self.client.post(
                "/upload",
                data={"file": (io.BytesIO(b"x" * (25 * 1024 * 1024 + 1)), "large.csv")},
                content_type="multipart/form-data",
            )
            self.assertEqual(oversized.status_code, 400)
            self.assertIn(b"File is too large", oversized.data)


if __name__ == "__main__":
    unittest.main()
