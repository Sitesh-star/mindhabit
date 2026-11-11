"""Build PDF reports from completed MindHabits analyses."""

from utils.pdf_report import create_wellness_pdf


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
