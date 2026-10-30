"""Build user-friendly CSV and PDF reports from completed MindHabits analyses."""

import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.pdf_report import create_wellness_pdf


DISCLAIMER = (
    "These results are for wellness awareness only and are not a medical diagnosis."
)


def _generated_label(generated_at=None):
    generated_at = generated_at or datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
    return generated_at.strftime("%d %b %Y, %I:%M %p %Z").strip()


def _main_metric(wellbeing_result):
    best_model = wellbeing_result.get("best_model", {})
    if wellbeing_result.get("problem_type") == "Classification":
        value = best_model.get("macro_f1")
        return "Macro F1", f"{value}%" if value is not None else "Not available"
    value = best_model.get("rmse")
    return "RMSE", str(value) if value is not None else "Not available"


def _csv_rows(dataset_name, wellbeing_result, habit_result, generated_at=None):
    rows = [
        ["Report Information", "Report Title", "MindHabits Analysis Report"],
        ["Report Information", "Generated", _generated_label(generated_at)],
        ["Report Information", "Dataset", dataset_name or "Not available"],
    ]

    if wellbeing_result:
        prediction = wellbeing_result.get("prediction", {})
        confidence = prediction.get("confidence")
        metric_name, metric_value = _main_metric(wellbeing_result)
        rows.extend([
            ["Mental Well-Being", "Analysed Outcome", wellbeing_result.get("outcome_display_name", "Not available")],
            ["Mental Well-Being", "Predicted Result", prediction.get("value", "Not available")],
            ["Mental Well-Being", "Prediction Confidence", f"{confidence}%" if confidence is not None else "Not available"],
            ["Mental Well-Being", "Confidence Level", prediction.get("confidence_band") or "Not available"],
            ["Mental Well-Being", "Best Model", wellbeing_result.get("best_model", {}).get("name", "Not available")],
            ["Mental Well-Being", metric_name, metric_value],
        ])
        for factor in wellbeing_result.get("contributing_factors", []):
            rows.append(["Important Factors", "Wellness Factor", factor.get("name", "")])
        for action in wellbeing_result.get("recommendations", []):
            rows.append(["Suggested Actions", "Action", action])

    if habit_result and habit_result.get("insights"):
        rows.extend([
            ["Habit Overview", "Positive Patterns", habit_result.get("positive_count", 0)],
            ["Habit Overview", "Areas to Improve", habit_result.get("improvement_count", 0)],
            ["Habit Overview", "Neutral / Monitor", habit_result.get("neutral_count", 0)],
            ["Habit Overview", "Total Insights", habit_result.get("total_insights", 0)],
        ])
        for insight in habit_result["insights"]:
            details = insight.get("summary", "")
            if insight.get("trend"):
                details = f"{details} Trend: {insight['trend']}."
            rows.extend([
                ["Habit Insights", f"{insight.get('badge', 'Insight')}: {insight.get('title', '')}", details],
                ["Habit Insights", f"Suggested Focus: {insight.get('title', '')}", insight.get("focus", "")],
            ])

    rows.append(["Disclaimer", "Wellness Awareness", DISCLAIMER])
    return rows


def create_csv_report(dataset_name, wellbeing_result, habit_result, generated_at=None):
    """Return a UTF-8 CSV report with clearly labelled result sections."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Section", "Item", "Details"])
    writer.writerows(
        _csv_rows(dataset_name, wellbeing_result, habit_result, generated_at)
    )
    return output.getvalue().encode("utf-8-sig")


def create_pdf_report(
    dataset_name,
    wellbeing_result,
    habit_result,
    generated_at=None,
    dataframe=None,
):
    """Return the professional report without rerunning any analysis models."""
    return create_wellness_pdf(
        dataset_name,
        wellbeing_result,
        habit_result,
        generated_at=generated_at,
        dataframe=dataframe,
    )
