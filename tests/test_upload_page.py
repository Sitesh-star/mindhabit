import shutil
import subprocess
import unittest
from pathlib import Path

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

        node = shutil.which("node")
        if node is not None:
            result = subprocess.run(
                [node, "--check", str(script_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
