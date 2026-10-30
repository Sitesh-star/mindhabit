import csv
from contextlib import closing
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import app as mindhabits
from utils.wellbeing_detector import detect_wellbeing_columns


class SpreadsheetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.previous_database = mindhabits.app.config["DATABASE"]
        self.previous_testing = mindhabits.app.config["TESTING"]
        self.previous_upload_folder = mindhabits.app.config["UPLOAD_FOLDER"]
        self.previous_database_url = mindhabits.app.config["DATABASE_URL"]
        mindhabits.app.config.update(
            TESTING=True, DATABASE=os.path.join(self.directory.name, "test.sqlite3"),
            DATABASE_URL=None, UPLOAD_FOLDER=self.directory.name,
        )
        self.client = mindhabits.app.test_client()
        self.payload = {
            "name": "My weekly tracker",
            "columns": ["Day", "Custom habit", "Notes"],
            "rows": [["Monday", "7.5", 'Comma, quote " and\nnewline'], ["Tuesday", "0", ""]],
        }

    def tearDown(self):
        mindhabits.app.config.update(
            DATABASE=self.previous_database, TESTING=self.previous_testing
        )
        mindhabits.app.config["UPLOAD_FOLDER"] = self.previous_upload_folder
        mindhabits.app.config["DATABASE_URL"] = self.previous_database_url

    def save(self):
        response = self.client.post("/api/spreadsheets", json=self.payload)
        self.assertEqual(response.status_code, 201)
        return response.get_json()

    def select_dataset(self, path):
        with self.client.session_transaction() as browser_session:
            browser_session["dataset_reference"] = {
                "storage": "local", "pathname": os.path.basename(path),
                "filename": os.path.basename(path),
            }

    def test_save_reopen_update_and_persistence(self):
        saved = self.save()
        with closing(sqlite3.connect(mindhabits.app.config["DATABASE"])) as database:
            record = database.execute(
                "SELECT name, rows_json FROM spreadsheets"
            ).fetchone()
        self.assertEqual(record[0], self.payload["name"])
        self.assertEqual(json.loads(record[1]), self.payload["rows"])
        reopened = self.client.get(saved["url"])
        self.assertEqual(reopened.status_code, 200)
        self.assertIn(b"My weekly tracker", reopened.data)
        self.payload["name"] = "Renamed tracker"
        self.payload["rows"][0][1] = "8.25"
        updated = self.client.put(saved["save_url"], json=self.payload)
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.get_json()["id"], saved["id"])
        self.assertIn(b"Renamed tracker", self.client.get("/manual").data)
        with closing(sqlite3.connect(mindhabits.app.config["DATABASE"])) as database:
            self.assertEqual(database.execute("SELECT COUNT(*) FROM spreadsheets").fetchone()[0], 1)

    def test_spreadsheets_are_isolated_by_visitor(self):
        saved = self.save()
        other_visitor = mindhabits.app.test_client()
        self.assertEqual(other_visitor.get(saved["url"]).status_code, 404)
        self.assertNotIn(b"My weekly tracker", other_visitor.get("/manual").data)

    def test_download_preserves_csv_values_and_unicode(self):
        self.payload["name"] = "Tracker / safe filename"
        self.payload["rows"][1][2] = "Mindfulness \u2713"
        saved = self.save()
        response = self.client.get(saved["download_url"])
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment;", response.headers["Content-Disposition"])
        self.assertIn("Tracker_safe_filename.csv", response.headers["Content-Disposition"])
        parsed = list(csv.reader(io.StringIO(response.data.decode("utf-8-sig"))))
        self.assertEqual(parsed, [self.payload["columns"], *self.payload["rows"]])

    def test_invalid_data_and_missing_spreadsheets(self):
        invalid = [None, [], {}, {**self.payload, "name": " "},
                   {**self.payload, "columns": [""]},
                   {**self.payload, "columns": ["Day", "day", "Notes"]},
                   {**self.payload, "rows": [["wrong length"]]},
                   {**self.payload, "rows": [[1, 2, 3]]}]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/spreadsheets", json=payload).status_code, 400)
        for path in ["/manual/999", "/manual/999/download"]:
            self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.put("/api/spreadsheets/999", json=self.payload).status_code, 404)

    def test_manual_data_uses_simplified_dashboard(self):
        self.payload = {
            "name": "Well-being tracker",
            "columns": ["Date", "Sleep Hours", "Mood"],
            "rows": [
                ["2026-10-24", "7", "Good"],
                ["2026-10-25", "8", "Good"],
                ["2026-10-26", "6", "Low"],
            ],
        }
        saved = self.save()
        response = self.client.get(saved["dashboard_url"])
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Well-being tracker", page)
        self.assertIn("Manual entry", page)
        self.assertIn("Quick summaries", page)
        self.assertIn("Distributions", page)
        self.assertIn("Average 7", page)
        self.assertIn("From 24 Oct 2026", page)
        self.assertNotIn("Dataset Columns", page)
        self.assertNotIn("Standard deviation", page)
        self.assertEqual(self.client.get("/dashboard").status_code, 200)

    def test_existing_routes_and_csv_upload(self):
        for route in ["/", "/upload", "/manual", "/habits", "/export", "/help"]:
            self.assertEqual(self.client.get(route).status_code, 200, route)
        no_data = self.client.get("/wellbeing")
        self.assertEqual(no_data.status_code, 302)
        self.assertTrue(no_data.location.endswith("/upload"))
        previous_directory = os.getcwd()
        try:
            os.chdir(self.directory.name)
            response = self.client.post("/upload", data={
                "file": (io.BytesIO(b"Day,Steps,Category\nMonday,5000,Walk\nTuesday,6000,Run\n"), "test.csv")
            }, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 302)
            self.assertTrue(response.location.endswith("/preview"))
            self.assertEqual(self.client.get("/preview").status_code, 200)
            self.assertEqual(self.client.get("/dashboard").status_code, 200)
            self.assertEqual(self.client.get("/wellbeing").status_code, 200)
        finally:
            os.chdir(previous_directory)

    def test_wellbeing_column_detection(self):
        dataset = os.path.join(self.directory.name, "wellbeing.csv")
        with open(dataset, "w", encoding="utf-8", newline="") as csv_file:
            csv_file.write(
                "User ID,Nightly Slept Hours,Felt Rested,Anxiety-Level,Steps That Day,"
                "Phone Usage Min,Coffee Cups,Meditation Minutes,Resting HR,"
                "Cognitive Performance Score,Favorite Color\n"
                "1,7.5,Yes,Low,5000,120,1,10,60,High,Blue\n"
                "2,6.0,No,High,3000,220,3,0,75,Low,Red\n"
                "3,8.0,Yes,Low,7000,90,1,15,58,High,Green\n"
                "4,7.0,Yes,Medium,6000,150,2,5,65,Medium,Blue\n"
            )
        self.select_dataset(dataset)

        page = self.client.get("/wellbeing")
        self.assertEqual(page.status_code, 200)
        for text in [
            b"Sleep", b"Stress / Mental State", b"Activity",
            b"Screen / Device Use", b"Caffeine / Lifestyle",
            b"Mindfulness / Recovery", b"Physiology",
            b"Cognitive / Performance", b"nightly_slept_hours",
            b"anxiety_level", b"resting_hr",
        ]:
            self.assertIn(text, page.data)
        self.assertIn(b"9 found", page.data)
        self.assertIn(b"felt_rested", page.data)
        self.assertIn(b"cognitive_performance_score", page.data)
        self.assertIn(b"Well-Being Analysis", page.data)
        self.assertIn(b"Analyse My Well-Being", page.data)
        self.assertIn(b"Detected Wellness Data", page.data)
        self.assertIn(b"Felt Rested", page.data)
        self.assertNotIn(b"Analysis type: Classification", page.data)
        self.assertIn(b"Other information considered", page.data)
        self.assertNotIn(b"user_id", page.data)
        self.assertNotIn(b"favorite_color", page.data)

        frame = mindhabits.pd.read_csv(dataset)
        frame.columns = frame.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
        detection = detect_wellbeing_columns(frame)
        self.assertEqual(detection["id_columns"], ["user_id"])
        resting_hr = next(column for column in detection["relevant_columns"] if column["name"] == "resting_hr")
        self.assertEqual(resting_hr["concept"], "Physiology")
        self.assertEqual(resting_hr["data_type"], "Number")
        self.assertEqual(resting_hr["unique_values"], 4)
        self.assertEqual(detection["recommended_outcome"]["name"], "felt_rested")
        self.assertEqual(detection["recommended_outcome"]["problem_type"], "Classification")
        self.assertEqual(
            [column["name"] for column in detection["alternative_outcomes"]],
            ["anxiety_level", "cognitive_performance_score"],
        )

    def test_wellbeing_outcome_ranking_supports_regression(self):
        frame = mindhabits.pd.DataFrame({
            "person_id": range(30),
            "cognitive_performance_score": [value % 15 for value in range(30)],
            "wellbeing_score": range(30),
            "sleep_duration": [6 + (value % 3) for value in range(30)],
        })

        detection = detect_wellbeing_columns(frame)
        self.assertEqual(detection["recommended_outcome"]["name"], "cognitive_performance_score")
        self.assertEqual(detection["recommended_outcome"]["display_name"], "Cognitive Performance Score")
        self.assertEqual(detection["recommended_outcome"]["problem_type"], "Regression")
        self.assertEqual(detection["alternative_outcomes"][0]["name"], "wellbeing_score")
        self.assertLess(
            detection["alternative_outcomes"][0]["score"],
            detection["recommended_outcome"]["score"],
        )

    def test_practical_wellness_outcomes_rank_above_clinical_columns(self):
        size = 30
        frame = mindhabits.pd.DataFrame({
            "person_id": range(size),
            "felt_rested": [value % 2 for value in range(size)],
            "sleep_disorder_risk": ["Low", "Medium", "High"] * 10,
            "sleep_quality_score": [5 + (value % 15) / 10 for value in range(size)],
            "stress_score": [value % 15 for value in range(size)],
            "cognitive_performance_score": [60 + value % 15 for value in range(size)],
            "mental_health_condition": ["Healthy", "Anxiety", "Depression"] * 10,
        })

        detection = detect_wellbeing_columns(frame)
        ranked_names = [column["name"] for column in detection["possible_outcomes"]]
        self.assertEqual(ranked_names, [
            "felt_rested",
            "sleep_disorder_risk",
            "sleep_quality_score",
            "stress_score",
            "cognitive_performance_score",
            "mental_health_condition",
        ])
        self.assertTrue(detection["possible_outcomes"][-1]["clinical"])
        self.assertEqual(detection["recommended_outcome"]["problem_type"], "Classification")

    def test_alternative_wellness_names_are_detected_flexibly(self):
        values = ["Low", "Medium", "High"] * 4
        for column_name in [
            "feeling_rested", "recovery_status", "mood_score",
            "wellbeing_score", "stress_level", "focus_score",
        ]:
            with self.subTest(column_name=column_name):
                detection = detect_wellbeing_columns(
                    mindhabits.pd.DataFrame({column_name: values})
                )
                self.assertEqual(
                    detection["recommended_outcome"]["name"], column_name
                )

    def test_wellbeing_reports_when_relevant_data_has_no_outcome(self):
        dataset = os.path.join(self.directory.name, "habits_only.csv")
        with open(dataset, "w", encoding="utf-8", newline="") as csv_file:
            csv_file.write("Steps,Sleep Duration\n5000,7\n6000,8\n5500,6\n")
        self.select_dataset(dataset)

        page = self.client.get("/wellbeing")
        self.assertEqual(page.status_code, 200)
        self.assertIn(
            b"MindHabits found relevant well-being information, but no suitable prediction outcome could be identified automatically.",
            page.data,
        )

    def test_wellbeing_shows_message_when_nothing_is_detected(self):
        dataset = os.path.join(self.directory.name, "unrelated.csv")
        with open(dataset, "w", encoding="utf-8", newline="") as csv_file:
            csv_file.write("Record ID,Favorite Color,City\n1,Blue,Ipoh\n2,Red,Johor\n3,Green,Klang\n")
        self.select_dataset(dataset)

        page = self.client.get("/wellbeing")
        self.assertEqual(page.status_code, 200)
        self.assertIn(
            b"No suitable mental well-being or habit-related variables were detected in this dataset.",
            page.data,
        )

    def test_rendered_javascript_syntax_and_safe_data(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js unavailable")
        self.payload["rows"][0][2] = "</script><script>alert('test')</script>"
        saved = self.save()
        for route in ["/manual", saved["url"]]:
            html = self.client.get(route).get_data(as_text=True)
            script = html.split("<script>", 1)[1].split("</script>", 1)[0]
            path = Path(self.directory.name) / "manual.js"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("</script><script>alert", html)


if __name__ == "__main__":
    unittest.main()
