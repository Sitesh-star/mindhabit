import unittest

import pandas as pd

from utils.recommendation_engine import generate_recommendations


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "sleep_hours": [4, 6, 6.5, 7, 7.2, 7.5, 8, 8.5],
            "sleep_duration_hrs": [4.2, 6, 6.4, 7, 7.1, 7.4, 8, 8.4],
            "screen_time_minutes": [600, 60, 90, 120, 180, 240, 300, 360],
            "daily_steps": [1000, 3000, 4500, 6000, 7500, 9000, 11000, 13000],
            "stress_score": [10, 1, 2, 3, 4, 5, 6, 7],
            "caffeine_mg": [500, 0, 40, 80, 120, 160, 200, 240],
        })
        self.factors = [
            {"name": "Sleep Duration", "column": "sleep_hours"},
            {"name": "Other Sleep Duration", "column": "sleep_duration_hrs"},
            {"name": "Screen Time", "column": "screen_time_minutes"},
            {"name": "Daily Steps", "column": "daily_steps"},
            {"name": "Stress Level", "column": "stress_score"},
            {"name": "Caffeine Intake", "column": "caffeine_mg"},
        ]

    def test_uses_record_values_and_deduplicates_concepts(self):
        recommendations = generate_recommendations(
            self.frame, 0, self.factors, limit=5
        )

        self.assertEqual(len(recommendations), 5)
        self.assertEqual(
            sum("sleep duration" in item.casefold() for item in recommendations),
            1,
        )
        self.assertTrue(any("screen use is relatively high" in item for item in recommendations))
        self.assertTrue(any("activity level is relatively low" in item for item in recommendations))
        self.assertTrue(any("stress level is relatively high" in item for item in recommendations))
        self.assertTrue(any("caffeine intake is relatively high" in item for item in recommendations))

    def test_positive_patterns_receive_maintenance_wording(self):
        recommendations = generate_recommendations(
            self.frame,
            7,
            [
                {"name": "Sleep Duration", "column": "sleep_hours"},
                {"name": "Daily Steps", "column": "daily_steps"},
            ],
        )

        self.assertEqual(len(recommendations), 2)
        self.assertTrue(all("maintain" in item.casefold() for item in recommendations))

    def test_low_caffeine_does_not_create_a_reduction_warning(self):
        recommendations = generate_recommendations(
            self.frame,
            1,
            [{"name": "Caffeine Intake", "column": "caffeine_mg"}],
        )

        self.assertEqual(recommendations, [])

    def test_missing_or_unknown_factors_are_ignored(self):
        recommendations = generate_recommendations(
            self.frame,
            0,
            [
                {"name": "User", "column": "user_id"},
                {"name": "Technical", "column": "unknown_field"},
            ],
        )

        self.assertEqual(recommendations, [])


if __name__ == "__main__":
    unittest.main()
