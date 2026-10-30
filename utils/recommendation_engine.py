"""Create simple, non-clinical habit suggestions from model-used wellness factors."""

import pandas as pd


CONCEPT_RULES = (
    ("sleep_duration", ("sleep_duration", "sleep_hours", "hours_slept", "slept_hours")),
    ("sleep_quality", ("sleep_quality", "quality_of_sleep", "sleep_efficiency")),
    ("sleep_latency", ("sleep_latency", "latency", "time_to_sleep")),
    ("wake_episodes", ("wake_episode", "night_waking", "awaken", "wake_after_sleep")),
    ("sleep_routine", ("bedtime", "sleep_consistency")),
    ("sleep_routine", ("sleep", "slept")),
    ("screen_use", ("screen", "phone", "device_usage", "device_use")),
    ("caffeine", ("caffeine", "coffee")),
    ("alcohol", ("alcohol",)),
    ("activity", ("steps", "activity", "exercise", "workout", "movement", "distance")),
    ("mindfulness", ("mindfulness", "meditation", "relaxation")),
    ("stress", ("stress", "anxiety")),
    ("hrv", ("hrv", "heart_rate_variability")),
    ("heart_rate", ("resting_heart_rate", "resting_hr", "heart_rate", "heartrate")),
    ("recovery", ("recovery", "rested", "refreshed", "rest", "fatigue", "tired")),
    ("mental_wellbeing", ("mood", "emotion", "wellbeing", "well_being")),
)


def _matches_keyword(column_name, keyword):
    name = str(column_name).casefold()
    keyword = keyword.casefold()
    if len(keyword) <= 4:
        return keyword in name.split("_")
    return keyword in name


def _detect_concept(column_name):
    for concept, keywords in CONCEPT_RULES:
        if any(_matches_keyword(column_name, keyword) for keyword in keywords):
            return concept
    return None


def _relative_position(series, current_value):
    """Return low, typical, or high without exposing the underlying quartiles."""
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    current_numeric = pd.to_numeric(pd.Series([current_value]), errors="coerce").iloc[0]
    if pd.notna(current_numeric) and len(numeric_series) >= 4:
        lower_quartile = numeric_series.quantile(0.25)
        upper_quartile = numeric_series.quantile(0.75)
        if lower_quartile < upper_quartile:
            if current_numeric <= lower_quartile:
                return "low"
            if current_numeric >= upper_quartile:
                return "high"
            return "typical"

    text = str(current_value).strip().casefold().replace("-", "_").replace(" ", "_")
    if text in {"low", "very_low", "poor", "inactive"}:
        return "low"
    if text in {"high", "very_high", "good", "excellent", "active"}:
        return "high"
    if text in {"medium", "moderate", "average", "typical"}:
        return "typical"
    return None


def _suggestion_for(concept, position, column_name):
    suggestions = {
        "sleep_duration": {
            "low": "Your sleep duration is relatively low. Consider allowing more time for sleep and keeping a consistent schedule.",
            "typical": "Your sleep duration is within the dataset's usual range. Try to maintain a consistent sleep schedule.",
            "high": "Your sleep duration is relatively strong. Try to maintain a consistent sleep schedule.",
            None: "Try to keep your sleep duration and sleep schedule consistent.",
        },
        "sleep_quality": {
            "low": "Your sleep quality is relatively low. Consider a regular wind-down routine and consistent sleep times.",
            "typical": "Your sleep quality is within the dataset's usual range. Maintain the routines that support it.",
            "high": "Your sleep quality is relatively strong. Try to maintain the routines that support it.",
            None: "A consistent wind-down routine may help support sleep quality.",
        },
        "sleep_latency": {
            "high": "It is taking relatively longer to settle to sleep. Consider a calm, screen-free wind-down routine.",
            "low": "Your time to settle to sleep is relatively low. Try to maintain your current wind-down routine.",
            "typical": "Keep a regular wind-down routine to support settling to sleep consistently.",
            None: "A calm, regular wind-down routine may support settling to sleep.",
        },
        "wake_episodes": {
            "high": "Night-time waking is relatively high. Consider a quieter, more consistent bedtime routine.",
            "low": "Night-time waking is relatively low. Try to maintain the routine that supports your rest.",
            "typical": "Keep sleep and wake times consistent to support uninterrupted rest.",
            None: "Consistent sleep and wake times may support more settled rest.",
        },
        "sleep_routine": {
            "low": "Your sleep routine measure is relatively low. Consider keeping bedtime and wake time more consistent.",
            "typical": "Keep bedtime and wake time consistent to support your current sleep routine.",
            "high": "Your sleep routine measure is relatively strong. Try to maintain this consistency.",
            None: "Consider keeping bedtime and wake time consistent across the week.",
        },
        "screen_use": {
            "high": "Your screen use is relatively high. Consider reducing device use before bedtime.",
            "low": "Your screen use is relatively low. Try to maintain this pattern, especially before bedtime.",
            "typical": "Consider keeping some screen-free time before bedtime.",
            None: "Consider keeping some screen-free time before bedtime.",
        },
        "caffeine": {
            "high": "Your caffeine intake is relatively high. Consider reducing late-day caffeine to support rest.",
            "typical": "Keep an eye on caffeine timing and consider avoiding it late in the day.",
            None: "Consider keeping caffeine earlier in the day when possible.",
        },
        "alcohol": {
            "high": "Your alcohol intake is relatively high. Consider limiting it, especially near bedtime.",
            "typical": "Consider keeping alcohol intake moderate and away from bedtime.",
            None: "Consider keeping alcohol intake moderate and away from bedtime.",
        },
        "activity": {
            "low": "Your activity level is relatively low. Consider adding light movement or exercise during the day.",
            "typical": "Your activity level is within the dataset's usual range. Keep building regular movement into your day.",
            "high": "Your activity level is relatively strong. Try to maintain this routine.",
            None: "Try to include regular light movement or exercise during the day.",
        },
        "mindfulness": {
            "low": "Mindfulness or relaxation time is relatively low. Consider starting with a short daily session.",
            "typical": "Continue making regular time for mindfulness or relaxation.",
            "high": "Your mindfulness or relaxation time is relatively strong. Try to maintain this routine.",
            None: "Consider adding a short mindfulness or relaxation session to your routine.",
        },
        "stress": {
            "high": "Your stress level is relatively high. Consider short breaks, relaxation, or mindfulness during the day.",
            "low": "Your stress level is relatively low. Try to maintain the routines that support this pattern.",
            "typical": "Regular short breaks and relaxation may help you manage everyday stress.",
            None: "Regular short breaks and relaxation may help support stress management.",
        },
        "hrv": {
            "low": "Your HRV is relatively low in this dataset. Consider consistent sleep, gentle activity, and recovery time.",
            "typical": "Continue balancing activity with consistent sleep and recovery time.",
            "high": "Your HRV is relatively strong in this dataset. Try to maintain your recovery routine.",
            None: "Consistent sleep, gentle activity, and recovery time may support your routine.",
        },
        "heart_rate": {
            "high": "Your heart rate measure is relatively high in this dataset. Consider allowing enough rest and recovery time.",
            "low": "Your heart rate measure is relatively low in this dataset. Continue monitoring the routines that support you.",
            "typical": "Continue balancing daily activity with enough rest and recovery.",
            None: "Balance daily activity with enough rest and recovery time.",
        },
        "mental_wellbeing": {
            "low": "Your well-being measure is relatively low. Consider small restorative breaks and supportive daily routines.",
            "typical": "Continue using regular routines that support your day-to-day well-being.",
            "high": "Your well-being measure is relatively strong. Try to maintain the routines that support it.",
            None: "Regular restorative breaks may help support day-to-day well-being.",
        },
    }

    if concept == "recovery":
        inverse_measure = any(
            keyword in str(column_name).casefold() for keyword in ("fatigue", "tired")
        )
        if inverse_measure and position == "high":
            return "Fatigue is relatively high. Consider allowing more time for rest and recovery."
        if inverse_measure and position == "low":
            return "Fatigue is relatively low. Try to maintain the routines that support your recovery."
        recovery_suggestions = {
            "low": "Your recovery measure is relatively low. Consider consistent sleep and more recovery time.",
            "typical": "Continue balancing activity with consistent rest and recovery.",
            "high": "Your recovery measure is relatively strong. Try to maintain this routine.",
            None: "Consistent sleep and recovery time may help support your routine.",
        }
        return recovery_suggestions[position]

    return suggestions.get(concept, {}).get(position)


def generate_recommendations(dataframe, record_index, important_factors, limit=5):
    """Return up to five unique suggestions for the analysed record."""
    if record_index not in dataframe.index:
        return []

    recommendations = []
    used_concepts = set()
    for factor in important_factors:
        column = factor.get("column")
        if column not in dataframe.columns:
            continue

        concept = _detect_concept(column)
        if concept is None or concept in used_concepts:
            continue

        current_value = dataframe.at[record_index, column]
        if pd.isna(current_value):
            continue

        position = _relative_position(dataframe[column], current_value)
        suggestion = _suggestion_for(concept, position, column)
        if not suggestion or suggestion in recommendations:
            continue

        recommendations.append(suggestion)
        used_concepts.add(concept)
        if len(recommendations) >= limit:
            break

    return recommendations
