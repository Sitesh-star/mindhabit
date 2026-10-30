"""Column-name based detection for well-being and habit data."""

from collections import OrderedDict

import pandas as pd


CONCEPT_GROUPS = OrderedDict([
    ("Sleep", (
        "sleep", "slept", "bedtime", "latency", "wake", "awake",
        "rested", "rem", "deep_sleep",
    )),
    ("Stress / Mental State", (
        "stress", "anxiety", "mood", "mental", "emotion", "wellbeing",
        "well_being", "depression", "diagnosis", "disorder", "disease",
    )),
    ("Activity", (
        "steps", "activity", "exercise", "workout", "movement", "distance",
    )),
    ("Screen / Device Use", (
        "screen", "phone", "device", "usage",
    )),
    ("Caffeine / Lifestyle", (
        "caffeine", "coffee", "alcohol",
    )),
    ("Mindfulness / Recovery", (
        "mindfulness", "meditation", "relaxation", "recovery",
    )),
    ("Physiology", (
        "heart_rate", "heartrate", "hr", "hrv", "spo2", "blood_pressure",
        "sbp", "dbp",
    )),
    ("Cognitive / Performance", (
        "cognitive", "performance", "concentration", "focus", "productivity",
    )),
])

OUTCOME_GROUPS = OrderedDict([
    ("Rest / Recovery", {
        "priority": 12,
        "keywords": (
            "rested", "rest", "recovery", "refreshed", "fatigue", "tired",
        ),
    }),
    ("Mood / Well-Being", {
        "priority": 11,
        "keywords": (
            "mood", "wellbeing", "well_being", "happiness", "emotional",
            "emotion",
        ),
    }),
    ("Sleep Outcome", {
        "priority": 10,
        "keywords": (
            "sleep_quality", "sleep_score", "sleep_risk", "sleep_disorder",
            "disorder_risk", "sleep_health",
        ),
    }),
    ("Stress", {
        "priority": 9,
        "keywords": (
            "stress", "stress_score", "stress_level", "perceived_stress",
            "anxiety", "distress",
        ),
    }),
    ("Cognitive / Performance", {
        "priority": 8,
        "keywords": (
            "cognitive", "performance", "focus", "concentration", "productivity",
        ),
    }),
    ("Clinical Diagnosis", {
        "priority": 4,
        "keywords": (
            "mental_health_condition", "diagnosis", "disorder_type", "disease",
            "depression_diagnosis", "anxiety_disorder",
        ),
    }),
])

CLINICAL_KEYWORDS = OUTCOME_GROUPS["Clinical Diagnosis"]["keywords"]
CLINICAL_PENALTY = 5

ID_NAMES = {
    "id", "user_id", "person_id", "record_id", "participant_id",
    "subject_id", "row_id", "uuid", "index", "record_number", "record_no",
}
ID_HINTS = {"id", "identifier", "uuid", "index", "code", "key"}


def _normalise_name(name):
    return str(name).strip().casefold().replace(" ", "_").replace("-", "_")


def _matches_keyword(column_name, keyword):
    """Match phrases flexibly while avoiding false matches for short terms like hr."""
    name = _normalise_name(column_name)
    keyword = _normalise_name(keyword)
    if len(keyword) <= 4:
        return keyword in name.split("_")
    return keyword in name


def _display_name(column_name):
    return _normalise_name(column_name).replace("_", " ").title()


def _problem_type(series, unique_values):
    if not pd.api.types.is_numeric_dtype(series) or unique_values <= 10:
        return "Classification"
    return "Regression"


def _friendly_data_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "Yes / No"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date"
    if pd.api.types.is_numeric_dtype(series):
        return "Number"
    return "Text / Category"


def _is_likely_id(column_name, series):
    name = _normalise_name(column_name)
    tokens = set(name.split("_"))

    if name in ID_NAMES or name.endswith("_id") or name.startswith("id_"):
        return True

    usable = series.dropna()
    if usable.empty:
        return False

    unique_ratio = usable.nunique(dropna=True) / len(usable)
    return unique_ratio >= 0.95 and bool(tokens & ID_HINTS)


def _score_outcome(column_info, series, total_rows):
    """Return outcome details and a readable suitability score, or None."""
    usable = series.dropna()
    usable_count = len(usable)
    unique_values = int(usable.nunique(dropna=True))
    completeness = usable_count / total_rows if total_rows else 0

    if usable_count < min(3, total_rows) or usable_count <= 1:
        return None
    if completeness < 0.3 or unique_values <= 1:
        return None

    matched_groups = []
    keyword_matches = 0
    for group_name, group in OUTCOME_GROUPS.items():
        matches = [
            keyword
            for keyword in group["keywords"]
            if _matches_keyword(column_info["name"], keyword)
        ]
        if matches:
            matched_groups.append((group_name, group["priority"]))
            keyword_matches += len(matches)

    if not matched_groups:
        return None

    matched_groups.sort(key=lambda match: -match[1])
    outcome_group, relevance_score = matched_groups[0]
    problem_type = _problem_type(series, unique_values)
    unique_ratio = unique_values / usable_count

    # Non-numeric near-unique labels behave like identifiers, not useful classes.
    if problem_type == "Classification" and unique_ratio >= 0.95:
        return None

    # Both analysis types can earn the same variation score. This avoids
    # preferring classification or regression just because of its type.
    if problem_type == "Classification":
        largest_class_share = float(usable.value_counts(normalize=True).iloc[0])
        if largest_class_share >= 0.98:
            return None
        if unique_values <= 10 and largest_class_share < 0.9:
            variation_score = 3
        elif unique_values <= 20:
            variation_score = 2
        else:
            variation_score = 1
    else:
        variation_score = 3 if unique_ratio <= 0.8 else 1

    if completeness >= 0.8:
        completeness_score = 3
    elif completeness >= 0.5:
        completeness_score = 2
    else:
        completeness_score = 1

    score = relevance_score
    score += 2  # The column is already relevant to well-being or habits.
    score += completeness_score
    score += variation_score
    score += min(max(keyword_matches - 1, 0), 1)  # Small specificity bonus.

    # Continuous scores may legitimately be unique, but rank them lower when
    # nearly every value differs because that pattern can also resemble an ID.
    if unique_ratio >= 0.98:
        score -= 5
    elif unique_ratio >= 0.95:
        score -= 3

    is_clinical = any(
        _matches_keyword(column_info["name"], keyword)
        for keyword in CLINICAL_KEYWORDS
    )
    if is_clinical:
        score -= CLINICAL_PENALTY

    outcome_info = dict(column_info)
    outcome_info.update({
        "outcome_group": outcome_group,
        "problem_type": problem_type,
        "score": score,
        "completeness": round(completeness * 100),
        "clinical": is_clinical,
    })
    return outcome_info


def detect_wellbeing_columns(df):
    """Return relevant concepts and ranked outcome candidates."""
    relevant_columns = []
    possible_outcomes = []
    id_columns = []

    for column in df.columns:
        series = df[column]

        if _is_likely_id(column, series):
            id_columns.append(column)
            continue

        concepts = [
            concept
            for concept, keywords in CONCEPT_GROUPS.items()
            if any(_matches_keyword(column, keyword) for keyword in keywords)
        ]
        if not concepts:
            continue

        usable_count = int(series.notna().sum())
        unique_values = int(series.nunique(dropna=True))
        column_info = {
            "name": column,
            "display_name": _display_name(column),
            "concept": concepts[0],
            "concepts": concepts,
            "data_type": _friendly_data_type(series),
            "unique_values": unique_values,
            "usable_values": usable_count,
        }
        relevant_columns.append(column_info)

        outcome_info = _score_outcome(column_info, series, len(df))
        if outcome_info:
            possible_outcomes.append(outcome_info)

    grouped_columns = OrderedDict()
    for concept in CONCEPT_GROUPS:
        matches = [
            column for column in relevant_columns if concept in column["concepts"]
        ]
        if matches:
            grouped_columns[concept] = matches

    possible_outcomes.sort(
        key=lambda candidate: (
            -candidate["score"],
            -candidate["completeness"],
        )
    )

    return {
        "relevant_columns": relevant_columns,
        "grouped_columns": grouped_columns,
        "possible_outcomes": possible_outcomes,
        "recommended_outcome": possible_outcomes[0] if possible_outcomes else None,
        "alternative_outcomes": possible_outcomes[1:],
        "id_columns": id_columns,
        "total_relevant_columns": len(relevant_columns),
    }
