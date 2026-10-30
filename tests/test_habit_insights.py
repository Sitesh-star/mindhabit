import os
import tempfile
import unittest

import pandas as pd

import app as mindhabits
from utils.habit_insights import analyze_habit_insights


class HabitInsightUtilityTests(unittest.TestCase):
    def test_dated_records_generate_directional_trends(self):
        size = 18
        frame = pd.DataFrame({
            "record_date": pd.date_range("2026-01-01", periods=size),
            "nightly_slept_hours": [5.5] * 6 + [6.5] * 6 + [8.0] * 6,
            "phone_use_minutes": [420] * 6 + [260] * 6 + [100] * 6,
            "coffee_mg": [60] * 6 + [120] * 6 + [260] * 6,
            "steps_that_day": [3000] * 6 + [6000] * 6 + [10000] * 6,
        })

        result = analyze_habit_insights(frame)

        self.assertEqual(result["date_column"], "record_date")
        self.assertEqual(result["total_insights"], 4)
        self.assertEqual(result["positive_count"], 3)
        self.assertEqual(result["improvement_count"], 1)
        summaries = " ".join(item["summary"] for item in result["insights"])
        self.assertIn("Later dated records", summaries)
        self.assertIn("more supportive direction", summaries)
        self.assertIn("less supportive direction", summaries)
        trends = {item["trend"] for item in result["insights"]}
        self.assertEqual(trends, {"Improving", "Worsening"})

    def test_distribution_analysis_works_without_dates_and_deduplicates(self):
        frame = pd.DataFrame({
            "sleep_duration_hours": [7, 7.1, 6.9, 7, 7.2, 6.8, 7.1, 7],
            "steps": [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000],
            "exercise_minutes": [10, 20, 30, 40, 50, 60, 30, 20],
            "caffeine_mg": [0, 10, 20, 30, 150, 250, 350, 450],
            "mindfulness_minutes": [0, 0, 0, 0, 0, 5, 10, 15],
        })

        result = analyze_habit_insights(frame)

        self.assertIsNone(result["date_column"])
        titles = [item["title"] for item in result["insights"]]
        self.assertIn("Sleep Consistency", titles)
        self.assertEqual(titles.count("Activity Pattern"), 1)
        self.assertIn("Caffeine Pattern", titles)
        self.assertIn("Mindfulness Practice", titles)
        self.assertGreaterEqual(result["positive_count"], 1)
        self.assertGreaterEqual(result["improvement_count"], 1)
        self.assertTrue(all(item["trend"] is None for item in result["insights"]))
        self.assertLessEqual(result["total_insights"], 5)
        display_titles = [item["display_title"] for item in result["insights"]]
        self.assertEqual(display_titles.count("Activity"), 1)
        self.assertTrue(all(item["display_status"] in {
            "Going well", "Needs attention", "Keep an eye on"
        } for item in result["insights"]))

    def test_stable_and_inconsistent_trends_are_only_used_with_dates(self):
        size = 12
        frame = pd.DataFrame({
            "date": pd.date_range("2026-02-01", periods=size),
            "screen_time_minutes": [120, 121, 119, 120, 122, 118] * 2,
            "stress_score": [10, 90, 15, 85, 20, 80] * 2,
        })

        result = analyze_habit_insights(frame)

        insights = {item["title"]: item for item in result["insights"]}
        self.assertEqual(insights["Screen Use Pattern"]["status"], "neutral")
        self.assertEqual(insights["Screen Use Pattern"]["trend"], "Stable")
        self.assertEqual(insights["Stress Pattern"]["status"], "improve")
        self.assertEqual(insights["Stress Pattern"]["trend"], "Inconsistent")
        self.assertEqual(result["neutral_count"], 1)

    def test_unrelated_or_too_small_data_returns_no_insights(self):
        frame = pd.DataFrame({
            "record_id": [1, 2, 3],
            "favourite_colour": ["blue", "red", "green"],
            "sleep": [7, 6, 8],
        })

        result = analyze_habit_insights(frame)

        self.assertEqual(result["total_insights"], 0)
        self.assertEqual(result["insights"], [])


class HabitInsightRouteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.previous_upload_folder = mindhabits.app.config["UPLOAD_FOLDER"]
        self.previous_testing = mindhabits.app.config["TESTING"]
        mindhabits.app.config["TESTING"] = True
        self.client = mindhabits.app.test_client()
        mindhabits.app.config["UPLOAD_FOLDER"] = self.directory.name

    def tearDown(self):
        mindhabits.app.config["UPLOAD_FOLDER"] = self.previous_upload_folder
        mindhabits.app.config["TESTING"] = self.previous_testing

    def select_dataset(self, path):
        with self.client.session_transaction() as browser_session:
            browser_session["dataset_reference"] = {
                "storage": "local", "pathname": os.path.basename(path),
                "filename": os.path.basename(path),
            }

    def test_page_prompts_for_upload_when_no_dataset_exists(self):
        with self.client.session_transaction() as browser_session:
            browser_session.pop("dataset_reference", None)

        response = self.client.get("/habits")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload data to explore your habit patterns", response.data)

    def test_page_renders_dataset_summary_and_cards(self):
        dataset = os.path.join(self.directory.name, "habits.csv")
        pd.DataFrame({
            "Date": pd.date_range("2026-01-01", periods=12),
            "Sleep Hours": [6, 6.2, 6.4, 6.5, 6.8, 7, 7.2, 7.4, 7.5, 7.7, 7.8, 8],
            "Daily Steps": [3000, 3500, 4000, 5000, 5500, 6000, 7000, 7500, 8000, 9000, 9500, 10000],
            "Screen Minutes": [400, 390, 370, 340, 320, 300, 270, 250, 220, 190, 160, 130],
        }).to_csv(dataset, index=False)
        self.select_dataset(dataset)

        response = self.client.get("/habits")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Your Habit Insights", response.data)
        self.assertIn(b"habits going well", response.data)
        self.assertIn(b"habits need attention", response.data)
        self.assertIn(b"habits to keep an eye on", response.data)
        self.assertIn(b"Keep it up", response.data)
        self.assertNotIn(b"Why it matters", response.data)
        self.assertNotIn(b"Suggested focus", response.data)
        self.assertNotIn(b"Total Insights", response.data)
        self.assertNotIn(b"records and", response.data)
        self.assertNotIn(b"sleep_hours", response.data)
        self.assertNotIn(b"daily_steps", response.data)

    def test_page_uses_clear_empty_pattern_message(self):
        dataset = os.path.join(self.directory.name, "unrelated.csv")
        pd.DataFrame({
            "Record ID": [1, 2, 3, 4, 5, 6],
            "Favourite Colour": ["Blue", "Red", "Green", "Blue", "Red", "Green"],
        }).to_csv(dataset, index=False)
        self.select_dataset(dataset)

        response = self.client.get("/habits")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"No clear habit patterns were found in this dataset.",
            response.data,
        )


if __name__ == "__main__":
    unittest.main()
