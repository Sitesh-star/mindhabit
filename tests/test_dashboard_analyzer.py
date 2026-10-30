import unittest

import pandas as pd

from utils.dashboard_analyzer import (
    build_dashboard_content,
    prepare_dashboard_dataframe,
)


class DashboardAnalyzerTests(unittest.TestCase):
    def test_preparsed_datetime_columns_remain_dates(self):
        frame = pd.DataFrame({
            "Record Date": pd.date_range("2026-01-01", periods=4),
            "Sleep Hours": [6, 7, 8, 7.5],
        })

        prepared = prepare_dashboard_dataframe(frame)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared["record_date"]))

    def test_alternative_wellness_names_are_cleaned_and_detected(self):
        size = 24
        frame = pd.DataFrame({
            " Person ID ": [f"P{index:03d}" for index in range(size)],
            "Record Date": pd.date_range("2026-01-01", periods=size),
            "Nightly-Slept Hrs": [6 + index % 4 * 0.5 for index in range(size)],
            "Tension Level": [index % 8 for index in range(size)],
            "Walking Distance": [2 + index % 10 for index in range(size)],
            "Phone Usage": [60 + index * 5 for index in range(size)],
            "Meditation Minutes": [index % 15 for index in range(size)],
            "Resting Pulse": [58 + index % 12 for index in range(size)],
            "Coffee Intake": [index % 4 for index in range(size)],
        })

        prepared = prepare_dashboard_dataframe(frame)
        result = build_dashboard_content(prepared)

        self.assertIn("nightly_slept_hrs", prepared.columns)
        self.assertTrue(result["wellness_detected"])
        self.assertLessEqual(len(result["column_summaries"]), 6)
        self.assertLessEqual(len(result["distribution_charts"]), 4)
        kinds = {summary["kind"] for summary in result["column_summaries"]}
        self.assertTrue({"Sleep", "Stress", "Activity"} <= kinds)
        self.assertNotIn(
            "person_id", {summary["name"] for summary in result["column_summaries"]}
        )
        self.assertTrue(any(chart["type"] == "line" for chart in result["distribution_charts"]))

    def test_best_column_is_selected_once_for_each_concept(self):
        frame = pd.DataFrame({
            "sleep_notes": [f"unique note {index}" for index in range(30)],
            "sleep_quality_rating": [None] * 18 + list(range(12)),
            "hours_slept": [6 + index % 4 * 0.5 for index in range(30)],
            "steps": [4000 + index * 100 for index in range(30)],
        })

        result = build_dashboard_content(prepare_dashboard_dataframe(frame))
        sleep_summaries = [
            summary for summary in result["column_summaries"]
            if summary["kind"] == "Sleep"
        ]

        self.assertEqual(len(sleep_summaries), 1)
        self.assertEqual(sleep_summaries[0]["name"], "hours_slept")

    def test_unrelated_data_uses_safe_generic_fallback(self):
        frame = pd.DataFrame({
            "Record ID": range(20),
            "Favourite Colour": ["Blue", "Red", "Green", "Blue"] * 5,
            "City": ["Ipoh", "Johor"] * 10,
            "Score Value": [10 + index % 5 for index in range(20)],
            "Unique Comment": [f"comment {index}" for index in range(20)],
        })

        result = build_dashboard_content(prepare_dashboard_dataframe(frame))

        self.assertFalse(result["wellness_detected"])
        names = {summary["name"] for summary in result["column_summaries"]}
        self.assertIn("score_value", names)
        self.assertNotIn("record_id", names)
        self.assertNotIn("unique_comment", names)
        self.assertLessEqual(len(result["column_summaries"]), 6)
        self.assertLessEqual(len(result["distribution_charts"]), 4)

    def test_categorical_data_without_dates_is_handled_safely(self):
        frame = pd.DataFrame({
            "Mood Response": ["Good", "Neutral", "Low"] * 5,
            "Exercise Type": ["Walk", "Run", "Rest"] * 5,
            "Meditation Done": ["Yes", "No", "Yes"] * 5,
        })

        result = build_dashboard_content(prepare_dashboard_dataframe(frame))

        self.assertIsNone(result["date_column"])
        self.assertTrue(result["wellness_detected"])
        self.assertTrue(result["distribution_charts"])
        self.assertTrue(all(
            chart["type"] == "bar" for chart in result["distribution_charts"]
        ))


if __name__ == "__main__":
    unittest.main()
