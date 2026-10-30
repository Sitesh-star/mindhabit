import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import app as mindhabits
from utils.ml_analyzer import (
    AnalysisError,
    _classification_models,
    _confidence_band,
    _factor_profile,
    analyze_models,
)


class ModelAnalyzerTests(unittest.TestCase):
    def test_classification_compares_four_models_with_group_split(self):
        size = 120
        frame = pd.DataFrame({
            "user_id": [f"U{index // 4:03d}" for index in range(size)],
            "sleep_hours": [6 + (index % 2) * 1.5 for index in range(size)],
            "steps": [4000 + (index % 12) * 300 for index in range(size)],
            "activity_type": ["Walk", "Run", "Rest"] * 40,
            "constant_field": [1] * size,
            "daily_note": [f"Unique note {index}" for index in range(size)],
            "mood_score": [index % 5 for index in range(size)],
            "felt_rested": [0, 1] * 60,
        })

        result = analyze_models(
            frame,
            target_column="felt_rested",
            problem_type="Classification",
            id_columns=["user_id"],
            leakage_columns=["mood_score"],
        )

        self.assertEqual(len(result["models"]), 4)
        self.assertEqual(
            {model["name"] for model in result["models"]},
            {
                "Logistic Regression", "Decision Tree Classifier",
                "Random Forest Classifier", "XGBoost Classifier",
            },
        )
        self.assertEqual(result["split_method"], "Grouped by User Id")
        self.assertEqual(result["train_rows"] + result["test_rows"], 119)
        self.assertIn("user_id", result["excluded_predictors"])
        self.assertIn("constant_field", result["excluded_predictors"])
        self.assertIn("daily_note", result["excluded_predictors"])
        self.assertIn("mood_score", result["excluded_predictors"])
        self.assertIn("macro_precision", result["best_model"])
        self.assertIn("macro_recall", result["best_model"])
        self.assertIn("weighted_f1", result["best_model"])
        expected_best = max(
            result["models"], key=lambda model: (model["macro_f1"], model["accuracy"])
        )
        self.assertEqual(result["best_model"], expected_best)
        self.assertIn(result["prediction"]["value"], {"No", "Yes"})
        self.assertIsNotNone(result["prediction"]["confidence"])
        self.assertIn(
            result["prediction"]["confidence_band"],
            {"Low confidence", "Moderate confidence", "Higher confidence"},
        )
        self.assertIn("Last usable record", result["prediction"]["record_description"])
        self.assertGreaterEqual(len(result["contributing_factors"]), 3)
        self.assertLessEqual(len(result["contributing_factors"]), 5)
        factor_names = {factor["name"] for factor in result["contributing_factors"]}
        self.assertTrue(factor_names <= {
            "Sleep Duration", "Daily Steps", "Physical Activity: Walk",
            "Physical Activity: Run", "Physical Activity: Rest",
        })
        self.assertFalse(any("User" in name or "Mood" in name for name in factor_names))
        self.assertTrue(result["recommendations"])
        self.assertLessEqual(len(result["recommendations"]), 5)

    def test_factor_labels_and_confidence_bands_are_user_friendly(self):
        self.assertEqual(_factor_profile("sleep_duration_hours")["label"], "Sleep Duration")
        self.assertEqual(_factor_profile("screen_time_before_bed_mins")["label"], "Screen Time Before Bed")
        self.assertEqual(_factor_profile("caffeine_mg")["label"], "Caffeine Intake")
        self.assertEqual(_factor_profile("hrv_rmssd_ms")["label"], "HRV")
        self.assertEqual(_factor_profile("stress_score")["label"], "Stress Level")
        self.assertEqual(_factor_profile("gender")["tier"], 3)
        self.assertIsNone(_factor_profile("measurement_date"))
        self.assertIsNone(_factor_profile("record_id"))

        self.assertIsNone(_confidence_band(None))
        self.assertEqual(_confidence_band(49.9), "Low confidence")
        self.assertEqual(_confidence_band(50), "Moderate confidence")
        self.assertEqual(_confidence_band(69.9), "Moderate confidence")
        self.assertEqual(_confidence_band(70), "Higher confidence")

    def test_xgboost_objective_changes_for_binary_and_multiclass(self):
        binary = next(
            model for name, model, _ in _classification_models(2)
            if name == "XGBoost Classifier"
        )
        multiclass = next(
            model for name, model, _ in _classification_models(4)
            if name == "XGBoost Classifier"
        )

        self.assertEqual(binary.get_params()["objective"], "binary:logistic")
        self.assertEqual(multiclass.get_params()["objective"], "multi:softprob")
        self.assertEqual(multiclass.get_params()["num_class"], 4)

    def test_regression_compares_four_models_with_standard_split(self):
        size = 120
        exercise = np.arange(size) % 20
        frame = pd.DataFrame({
            "record_id": range(size),
            "measurement_date": pd.date_range("2026-01-01", periods=size),
            "exercise_minutes": exercise,
            "sleep_hours": 6 + (np.arange(size) % 5) * 0.4,
            "day_type": ["Weekday", "Weekend"] * 60,
            "wellbeing_score": 40 + exercise * 1.8 + (np.arange(size) % 3),
        })

        result = analyze_models(
            frame,
            target_column="wellbeing_score",
            problem_type="Regression",
            id_columns=["record_id"],
        )

        self.assertEqual(len(result["models"]), 4)
        self.assertEqual(result["split_method"], "Standard 80/20 split")
        self.assertEqual(result["train_rows"] + result["test_rows"], 119)
        self.assertIn("mae", result["best_model"])
        self.assertIn("rmse", result["best_model"])
        self.assertIn("r2", result["best_model"])
        expected_best = min(
            result["models"], key=lambda model: (model["rmse"], -model["r2"])
        )
        self.assertEqual(result["best_model"], expected_best)
        self.assertIn("30 Apr 2026", result["prediction"]["record_description"])
        self.assertIsNone(result["prediction"]["confidence"])
        self.assertIsNone(result["prediction"]["confidence_band"])
        self.assertRegex(result["prediction"]["value"], r"^-?\d+(\.\d+)?$")
        self.assertTrue(result["contributing_factors"])
        factor_names = {factor["name"] for factor in result["contributing_factors"]}
        self.assertIn("Exercise", factor_names)
        self.assertIn("Sleep Duration", factor_names)
        self.assertFalse(any("Date" in name or "Record" in name for name in factor_names))

    def test_unsuitable_datasets_fail_with_friendly_errors(self):
        too_small = pd.DataFrame({
            "sleep_hours": range(10),
            "mood": ["Good", "Low"] * 5,
        })
        with self.assertRaisesRegex(AnalysisError, "At least 20 usable rows"):
            analyze_models(too_small, "mood", "Classification")

        constant_target = pd.DataFrame({
            "sleep_hours": range(30),
            "mood": ["Good"] * 30,
        })
        with self.assertRaisesRegex(AnalysisError, "only one category"):
            analyze_models(constant_target, "mood", "Classification")

        no_predictors = pd.DataFrame({
            "user_id": range(30),
            "felt_rested": ["Yes", "No"] * 15,
        })
        with self.assertRaisesRegex(AnalysisError, "No usable predictor columns"):
            analyze_models(
                no_predictors, "felt_rested", "Classification", ["user_id"]
            )


class WellbeingRouteAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.previous_upload_folder = mindhabits.app.config["UPLOAD_FOLDER"]
        self.previous_testing = mindhabits.app.config["TESTING"]
        mindhabits.app.config["TESTING"] = True
        self.dataset = os.path.join(self.directory.name, "wellbeing.csv")
        pd.DataFrame({
            "sleep_hours": [6, 7, 8, 7] * 8,
            "steps": [4000, 5000, 6000, 7000] * 8,
            "felt_rested": ["No", "Yes", "Yes", "No"] * 8,
        }).to_csv(self.dataset, index=False)
        self.client = mindhabits.app.test_client()
        mindhabits.app.config["UPLOAD_FOLDER"] = self.directory.name
        with self.client.session_transaction() as browser_session:
            browser_session["dataset_reference"] = {
                "storage": "local", "pathname": os.path.basename(self.dataset),
                "filename": os.path.basename(self.dataset),
            }

    def tearDown(self):
        mindhabits.app.config["UPLOAD_FOLDER"] = self.previous_upload_folder
        mindhabits.app.config["TESTING"] = self.previous_testing

    def test_models_run_only_when_button_posts(self):
        fake_result = {
            "outcome_display_name": "Felt Rested",
            "problem_type": "Classification",
            "best_model": {
                "name": "Logistic Regression", "accuracy": 75.0,
                "macro_precision": 74.0, "macro_recall": 73.0,
                "macro_f1": 72.0, "weighted_f1": 74.0,
            },
            "models": [{
                "name": "Logistic Regression", "accuracy": 75.0,
                "macro_precision": 74.0, "macro_recall": 73.0,
                "macro_f1": 72.0, "weighted_f1": 74.0,
            }],
            "prediction": {
                "outcome_label": "Predicted Felt Rested",
                "value": "Yes",
                "confidence": 44.8,
                "confidence_band": "Low confidence",
                "record_description": "Last usable record in the dataset (row 32)",
            },
            "contributing_factors": [
                {"name": "Sleep Duration", "column": "sleep_hours"},
                {"name": "Sleep Duration", "column": "nightly_sleep"},
                {"name": "Daily Steps", "column": "steps"},
            ],
            "recommendations": [
                "Try to keep your sleep duration and sleep schedule consistent.",
                "Try to include regular light movement or exercise during the day.",
            ],
            "rows_used": 32,
            "predictor_count": 2,
            "split_method": "Standard 80/20 split",
            "sampled": False,
            "model_errors": [],
        }

        with patch("app.analyze_models", return_value=fake_result) as analyzer:
            get_response = self.client.get("/wellbeing")
            self.assertEqual(get_response.status_code, 200)
            self.assertIn(b"Analyse My Well-Being", get_response.data)
            self.assertIn(b"Detected Wellness Data", get_response.data)
            self.assertNotIn(b'<details class="wellbeing-disclosure detection-disclosure" open', get_response.data)
            analyzer.assert_not_called()

            post_response = self.client.post("/wellbeing")
            self.assertEqual(post_response.status_code, 200)
            self.assertIn(b"the pattern in your recent data is closest to", post_response.data)
            self.assertIn(b"Technical Details", post_response.data)
            self.assertNotIn(b'<details class="wellbeing-disclosure technical-details-panel" open', post_response.data)
            self.assertIn(b"Logistic Regression", post_response.data)
            self.assertIn(b"44.8%", post_response.data)
            self.assertIn(b"Low confidence", post_response.data)
            self.assertIn(b"general wellness guidance", post_response.data)
            self.assertIn(b"Factors Related to This Result", post_response.data)
            self.assertIn(b"do not necessarily mean they caused", post_response.data)
            self.assertEqual(post_response.data.count(b"Sleep Duration"), 1)
            self.assertIn(b"Suggested Actions", post_response.data)
            self.assertLess(
                post_response.data.index(b"Suggested Actions"),
                post_response.data.index(b"Technical Details"),
            )
            analyzer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
