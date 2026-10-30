import math
import os
import sqlite3
import tempfile
import unittest

import numpy as np

import app as mindhabits
from utils.database import Database, json_safe


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(
            f"sqlite:///{os.path.join(self.directory.name, 'persistence.sqlite3')}"
        )

    def test_postgresql_configuration_uses_server_placeholders(self):
        database = Database("postgresql://user:secret@example.test/mindhabits")
        self.assertTrue(database.postgres)
        self.assertEqual(database.placeholder, "%s")

    def test_spreadsheet_crud_and_visitor_isolation(self):
        spreadsheet_id = self.database.create_spreadsheet(
            "visitor-a", "Tracker", ["Sleep"], [["7"]]
        )
        self.assertEqual(
            self.database.get_spreadsheet("visitor-a", spreadsheet_id)["rows"],
            [["7"]],
        )
        self.assertIsNone(self.database.get_spreadsheet("visitor-b", spreadsheet_id))
        self.assertEqual(self.database.list_spreadsheets("visitor-b"), [])
        self.assertTrue(self.database.update_spreadsheet(
            "visitor-a", spreadsheet_id, "Updated", ["Sleep"], [["8"]]
        ))
        self.assertFalse(self.database.update_spreadsheet(
            "visitor-b", spreadsheet_id, "Other", ["Sleep"], [["1"]]
        ))
        self.assertEqual(
            self.database.get_spreadsheet("visitor-a", spreadsheet_id)["name"],
            "Updated",
        )

    def test_legacy_local_schema_is_migrated_without_blocking_new_saves(self):
        path = os.path.join(self.directory.name, "legacy.sqlite3")
        connection = sqlite3.connect(path)
        connection.execute("""
            CREATE TABLE spreadsheets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute(
            "INSERT INTO spreadsheets (name, data) VALUES (?, ?)",
            ("Legacy", '{"columns":["Sleep"],"rows":[["7"]]}'),
        )
        connection.commit()
        connection.close()
        database = Database(f"sqlite:///{path}")
        saved_id = database.create_spreadsheet(
            "visitor-a", "New", ["Mood"], [["Good"]]
        )
        self.assertEqual(database.get_spreadsheet("visitor-a", saved_id)["name"], "New")
        self.assertEqual(database.list_spreadsheets("visitor-a")[1]["name"], "Legacy")

    def test_analysis_is_persisted_by_visitor_and_dataset(self):
        result = {"prediction": np.int64(1), "confidence": np.float64(72.5)}
        self.database.save_analysis_result(
            "visitor-a", "blob:datasets/a.csv", "wellbeing", result
        )
        stored = self.database.get_analysis_results(
            "visitor-a", "blob:datasets/a.csv"
        )
        self.assertEqual(stored["wellbeing"]["prediction"], 1)
        self.assertEqual(stored["wellbeing"]["confidence"], 72.5)
        self.assertIsNone(self.database.get_analysis_results(
            "visitor-a", "blob:datasets/b.csv"
        )["wellbeing"])
        self.assertIsNone(self.database.get_analysis_results(
            "visitor-b", "blob:datasets/a.csv"
        )["wellbeing"])

    def test_json_safe_handles_non_finite_and_numpy_values(self):
        converted = json_safe({"nan": math.nan, "number": np.int64(4)})
        self.assertIsNone(converted["nan"])
        self.assertEqual(converted["number"], 4)

    def test_visitor_id_is_created_and_process_globals_are_removed(self):
        client = mindhabits.app.test_client()
        client.get("/")
        with client.session_transaction() as browser_session:
            self.assertRegex(browser_session["visitor_id"], r"^[0-9a-f-]{36}$")
        self.assertFalse(hasattr(mindhabits, "current_spreadsheet_id"))
        self.assertFalse(hasattr(mindhabits, "latest_analysis_results"))


if __name__ == "__main__":
    unittest.main()
