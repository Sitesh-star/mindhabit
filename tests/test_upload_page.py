import os
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
            completed = self.client.post("/upload/complete", json={
                "storage": "blob",
                "pathname": metadata["pathname"],
                "filename": metadata["filename"],
            })

        self.assertEqual(completed.status_code, 200)
        with self.client.session_transaction() as browser_session:
            reference = browser_session["dataset_reference"]
        self.assertEqual(reference["storage"], "blob")
        self.assertEqual(reference["pathname"], metadata["pathname"])
        self.assertEqual(reference["filename"], "wellness_data.csv")


if __name__ == "__main__":
    unittest.main()
