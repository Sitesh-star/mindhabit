import csv
import io
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

import app as mindhabits
from utils.export_manager import create_csv_report, create_pdf_report
from utils.pdf_report import build_report_content


def sample_wellbeing_result():
    return {
        "outcome_display_name": "Felt Rested",
        "problem_type": "Classification",
        "prediction": {
            "value": "Yes",
            "confidence": 78.4,
            "confidence_band": "Higher confidence",
        },
        "best_model": {
            "name": "Random Forest Classifier",
            "macro_f1": 81.2,
            "accuracy": 82.0,
        },
        "contributing_factors": [
            {"name": "Sleep Duration", "column": "sleep_hours"},
            {"name": "Stress Level", "column": "stress_score"},
        ],
        "recommendations": [
            "Try to maintain a consistent sleep schedule.",
            "Consider regular relaxation breaks.",
        ],
    }


def sample_habit_result():
    insights = [
        {
            "title": "Sleep Consistency",
            "badge": "Positive Pattern",
            "status": "positive",
            "summary": "Sleep duration remains relatively consistent.",
            "focus": "Keep maintaining regular sleep and wake times.",
            "trend": "Stable",
        },
        {
            "title": "Caffeine Pattern",
            "badge": "Suggested Focus",
            "status": "improve",
            "summary": "Caffeine intake is inconsistent.",
            "focus": "Consider keeping caffeine earlier in the day.",
            "trend": None,
        },
        {
            "title": "Stress Pattern",
            "badge": "Neutral / Monitor",
            "status": "neutral",
            "summary": "Stress remains stable across the records.",
            "focus": "Continue monitoring this pattern.",
            "trend": "Stable",
        },
    ]
    return {
        "insights": insights,
        "positive_count": 1,
        "improvement_count": 1,
        "neutral_count": 1,
        "total_insights": 3,
    }


class ExportManagerTests(unittest.TestCase):
    def setUp(self):
        self.generated_at = datetime(2026, 10, 27, 14, 30, tzinfo=timezone.utc)

    def test_csv_contains_clear_report_sections(self):
        report = create_csv_report(
            "wellness.csv",
            sample_wellbeing_result(),
            sample_habit_result(),
            self.generated_at,
        )
        rows = list(csv.reader(io.StringIO(report.decode("utf-8-sig"))))

        self.assertEqual(rows[0], ["Section", "Item", "Details"])
        sections = {row[0] for row in rows[1:]}
        self.assertTrue({
            "Mental Well-Being", "Important Factors", "Suggested Actions",
            "Habit Overview", "Habit Insights", "Disclaimer",
        } <= sections)
        text = "\n".join(",".join(row) for row in rows)
        self.assertIn("Felt Rested", text)
        self.assertIn("Higher confidence", text)
        self.assertIn("Positive Pattern: Sleep Consistency", text)
        self.assertNotIn("sleep_hours", text)

    def test_pdf_is_valid_and_contains_readable_sections(self):
        report = create_pdf_report(
            "wellness.csv",
            sample_wellbeing_result(),
            sample_habit_result(),
            self.generated_at,
        )

        self.assertTrue(report.startswith(b"%PDF-1.4"))
        self.assertTrue(report.endswith(b"%%EOF"))
        self.assertIn(b"MindHabits Wellness & Habit Report", report)
        self.assertIn(b"YOUR WELL-BEING RESULT", report)
        self.assertIn(b"Key Wellness Factors", report)
        self.assertIn(b"Your Habit Overview", report)
        self.assertIn(b"Your Recommended Actions", report)
        self.assertIn(b"What To Focus On Next", report)
        self.assertIn(b"Wellness Disclaimer", report)
        self.assertIn(b"Page 1 of", report)

    def test_pdf_content_uses_dataset_for_metadata_graphs_and_factor_context(self):
        frame = pd.DataFrame({
            "record_date": pd.date_range("2026-01-01", periods=12),
            "user_id": range(1, 13),
            "sleep_hours": [6, 6.2, 6.4, 6.5, 6.8, 7, 7.1, 7.3, 7.5, 7.7, 7.8, 8],
            "stress_score": [8, 7, 8, 6, 7, 6, 5, 5, 4, 4, 3, 3],
            "daily_steps": [3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500],
            "screen_minutes": [300, 290, 280, 260, 250, 240, 220, 210, 200, 190, 180, 170],
        })

        content = build_report_content(
            "wellness.csv",
            sample_wellbeing_result(),
            sample_habit_result(),
            self.generated_at,
            frame,
        )

        self.assertEqual(content["records"], 12)
        self.assertEqual(content["date_range"], "01 Jan 2026 to 12 Jan 2026")
        self.assertGreaterEqual(len(content["charts"]), 3)
        self.assertLessEqual(len(content["charts"]), 4)
        self.assertTrue(all(chart["concept"] for chart in content["charts"]))
        self.assertTrue(all(chart["explanation"] for chart in content["charts"]))
        self.assertNotIn("user_id", {chart["name"] for chart in content["charts"]})
        self.assertIn("latest recorded value", content["factors"][0]["pattern"].casefold())
        concepts = [item["concept"] for item in content["recommendations"]]
        self.assertEqual(len(concepts), len(set(concepts)))
        self.assertLessEqual(len(content["recommendations"]), 5)
        self.assertTrue(all(2 <= len(item["actions"]) <= 3 for item in content["recommendations"]))

        report = create_pdf_report(
            "wellness.csv",
            sample_wellbeing_result(),
            sample_habit_result(),
            self.generated_at,
            dataframe=frame,
        )
        self.assertIn(b"Important Graphs", report)
        self.assertIn(b"RECORDS ANALYSED", report)
        self.assertIn(b"01 Jan 2026 to 12 Jan 2026", report)


class ExportRouteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.previous_upload_folder = mindhabits.app.config["UPLOAD_FOLDER"]
        self.previous_database = mindhabits.app.config["DATABASE"]
        self.previous_database_url = mindhabits.app.config["DATABASE_URL"]
        self.previous_testing = mindhabits.app.config["TESTING"]
        mindhabits.app.config["TESTING"] = True
        mindhabits.app.config.update(
            DATABASE=os.path.join(self.directory.name, "test.sqlite3"),
            DATABASE_URL=None,
        )
        self.dataset = os.path.join(self.directory.name, "wellness.csv")
        with open(self.dataset, "w", encoding="utf-8") as dataset_file:
            dataset_file.write("sleep_hours,mood\n7,good\n")
        mindhabits.app.config["UPLOAD_FOLDER"] = self.directory.name
        self.client = mindhabits.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["visitor_id"] = "export-test-visitor"
            browser_session["dataset_reference"] = {
                "storage": "local", "pathname": os.path.basename(self.dataset),
                "filename": os.path.basename(self.dataset),
            }

    def tearDown(self):
        mindhabits.app.config["UPLOAD_FOLDER"] = self.previous_upload_folder
        mindhabits.app.config.update(
            DATABASE=self.previous_database, DATABASE_URL=self.previous_database_url
        )
        mindhabits.app.config["TESTING"] = self.previous_testing

    def test_empty_export_page_does_not_offer_an_empty_download(self):
        response = self.client.get("/export")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Run Mental Well-Being Analysis first to include prediction results in the report.",
            response.data,
        )
        self.assertIn(
            b"No habit insights are currently available for export.",
            response.data,
        )
        self.assertNotIn(b"Download CSV", response.data)
        self.assertEqual(self.client.get("/export/csv").status_code, 302)
        self.assertEqual(self.client.get("/export/pdf").status_code, 302)

    def test_cached_results_render_and_download_without_analysis(self):
        self.store_result("wellbeing", sample_wellbeing_result())
        self.store_result("habits", sample_habit_result())

        with (
            patch("app.analyze_models") as model_analyzer,
            patch("app.analyze_habit_insights") as habit_analyzer,
        ):
            page = self.client.get("/export")
            csv_response = self.client.get("/export/csv")
            pdf_response = self.client.get("/export/pdf")

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Felt Rested", page.data)
        self.assertIn(b"Sleep Consistency", page.data)
        self.assertIn(b"Download CSV", page.data)
        self.assertIn(b"Download PDF", page.data)
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(csv_response.mimetype, "text/csv")
        self.assertIn("mindhabits_analysis_report.csv", csv_response.headers["Content-Disposition"])
        self.assertIn(b"Important Factors", csv_response.data)
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response.mimetype, "application/pdf")
        self.assertIn("mindhabits_analysis_report.pdf", pdf_response.headers["Content-Disposition"])
        self.assertTrue(pdf_response.data.startswith(b"%PDF-1.4"))
        model_analyzer.assert_not_called()
        habit_analyzer.assert_not_called()

    def test_cache_is_ignored_for_a_different_dataset(self):
        with self.client.session_transaction() as browser_session:
            visitor_id = browser_session["visitor_id"]
        mindhabits._database().save_analysis_result(
            visitor_id, "local:old.csv", "wellbeing", sample_wellbeing_result()
        )

        response = self.client.get("/export")

        self.assertIn(b"No completed results are available yet", response.data)
        self.assertNotIn(b"Felt Rested", response.data)

    def store_result(self, result_type, result):
        with self.client.session_transaction() as browser_session:
            visitor_id = browser_session["visitor_id"]
        mindhabits._database().save_analysis_result(
            visitor_id,
            f"local:{os.path.basename(self.dataset)}",
            result_type,
            result,
        )


if __name__ == "__main__":
    unittest.main()
