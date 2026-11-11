"""Flexible, schema-independent summaries for the MindHabits Dashboard."""

import math
import re
from collections import OrderedDict

import pandas as pd

from utils.wellbeing_detector import ID_HINTS, ID_NAMES


CONCEPT_RULES = OrderedDict([
    ("Sleep", ("sleep", "slept", "sleeping", "bedtime", "rested", "latency", "wake")),
    ("Stress", ("stress", "anxiety", "tension")),
    ("Activity", ("steps", "activity", "exercise", "workout", "movement", "walking", "distance")),
    ("Screen Use", ("screen", "phone", "device_use", "digital")),
    ("Mindfulness", ("mindfulness", "meditation", "relaxation")),
    ("Mood / Well-Being", ("mood", "emotion", "wellbeing", "well_being", "happiness", "rested")),
    ("Physiology", ("heart_rate", "heartrate", "hr", "hrv", "spo2", "pulse", "blood_pressure", "sbp", "dbp")),
    ("Caffeine", ("caffeine", "coffee")),
    ("Alcohol", ("alcohol",)),
    ("Recovery", ("recovery", "readiness")),
])

CONCEPT_PRIORITY = {
    "Sleep": 8,
    "Stress": 7,
    "Activity": 7,
    "Screen Use": 6,
    "Mindfulness": 6,
    "Mood / Well-Being": 7,
    "Physiology": 6,
    "Caffeine": 5,
    "Alcohol": 5,
    "Recovery": 6,
}

TECHNICAL_TOKENS = {
    "identifier", "index", "uuid", "serial", "token", "hash", "key",
    "filename", "filepath", "url", "version", "id", "code", "encoded",
    "country", "region", "model",
}


def clean_column_names(columns):
    """Normalise names and keep collisions unique after cleaning."""
    cleaned = []
    seen = {}
    for position, column in enumerate(columns, start=1):
        name = str(column).strip().casefold()
        name = re.sub(r"[\s-]+", "_", name).strip("_")
        name = re.sub(r"_+", "_", name) or f"column_{position}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        cleaned.append(name if count == 1 else f"{name}_{count}")
    return cleaned


def prepare_dashboard_dataframe(dataframe):
    dataframe = dataframe.copy()
    dataframe.columns = clean_column_names(dataframe.columns)
    dataframe = dataframe.replace(r"^\s*$", pd.NA, regex=True)

    for column in dataframe.columns:
        if (
            pd.api.types.is_numeric_dtype(dataframe[column])
            or pd.api.types.is_datetime64_any_dtype(dataframe[column])
        ):
            continue
        present = dataframe[column].dropna()
        converted = pd.to_numeric(present, errors="coerce")
        if len(present) and converted.notna().all():
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe.replace([float("inf"), float("-inf")], float("nan"))


def _normalised(name):
    return str(name).strip().casefold().replace(" ", "_").replace("-", "_")


def _keyword_match(column_name, keyword):
    name = _normalised(column_name)
    keyword = _normalised(keyword)
    if len(keyword) <= 4:
        return keyword in name.split("_")
    return keyword in name


def _display_name(column_name):
    return _normalised(column_name).replace("_", " ").title()


def _is_identifier_or_technical(column_name, series):
    name = _normalised(column_name)
    tokens = set(name.split("_"))
    if name in ID_NAMES or name.endswith("_id") or name.startswith("id_"):
        return True
    if name.startswith("unnamed") or tokens & TECHNICAL_TOKENS:
        return True
    if "device_model" in name or "record_number" in name:
        return True

    usable = series.dropna()
    if usable.empty:
        return False
    unique_ratio = usable.nunique(dropna=True) / len(usable)
    if unique_ratio >= 0.95 and tokens & ID_HINTS:
        return True
    return False


def _find_date_column(dataframe):
    best = None
    total_rows = len(dataframe)
    for column in dataframe.columns:
        name = _normalised(column)
        tokens = set(name.split("_"))
        if "date" not in tokens and "timestamp" not in name:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            continue
        dates = pd.to_datetime(dataframe[column], errors="coerce", format="mixed", utc=True)
        usable = int(dates.notna().sum())
        if usable < min(3, total_rows) or not total_rows or usable / total_rows < 0.5:
            continue
        score = usable / total_rows
        if best is None or score > best[2]:
            best = (column, dates, score)
    return best


def _column_quality(column_name, series, concept=None):
    usable = series.dropna()
    if usable.empty:
        return None

    total = len(series)
    usable_count = len(usable)
    unique_count = int(usable.nunique(dropna=True))
    numeric = pd.api.types.is_numeric_dtype(series)

    if not numeric and unique_count > 40:
        return None

    score = (usable_count / total * 5) if total else 0
    score += 3 if numeric else 2
    if numeric:
        numeric_values = pd.to_numeric(usable, errors="coerce").dropna()
        if numeric_values.nunique() > 2:
            score += 2
    elif 2 <= unique_count <= 12:
        score += 2
    if concept:
        matches = [
            keyword for keyword in CONCEPT_RULES[concept]
            if _keyword_match(column_name, keyword)
        ]
        score += min(3, max((len(keyword) for keyword in matches), default=0) / 5)
        score += CONCEPT_PRIORITY[concept] / 10
    return score


def _detect_candidates(dataframe, date_column):
    candidates = []
    for column in dataframe.columns:
        if column == date_column:
            continue
        series = dataframe[column]
        if _is_identifier_or_technical(column, series):
            continue
        matched_concepts = [
            concept for concept, keywords in CONCEPT_RULES.items()
            if any(_keyword_match(column, keyword) for keyword in keywords)
        ]
        for concept in matched_concepts:
            quality = _column_quality(column, series, concept)
            if quality is not None:
                candidates.append({
                    "column": column,
                    "concept": concept,
                    "score": quality,
                })
    return candidates


def _select_columns(dataframe, date_column, limit=6):
    candidates = _detect_candidates(dataframe, date_column)
    selected = []
    used_columns = set()
    for concept in CONCEPT_RULES:
        concept_candidates = sorted(
            (item for item in candidates if item["concept"] == concept),
            key=lambda item: (-item["score"], item["column"]),
        )
        candidate = next(
            (item for item in concept_candidates if item["column"] not in used_columns),
            None,
        )
        if candidate:
            selected.append(candidate)
            used_columns.add(candidate["column"])

    selected.sort(
        key=lambda item: (-(item["score"] + CONCEPT_PRIORITY[item["concept"]]), item["column"])
    )
    selected = selected[:limit]
    wellness_count = len(selected)

    return selected[:limit], wellness_count, len(candidates)


def _number(value):
    value = round(float(value), 2)
    return f"{int(value):,}" if value.is_integer() else f"{value:,}"


def _unit_for(column_name):
    name = _normalised(column_name)
    tokens = set(name.split("_"))
    if tokens & {"minutes", "minute", "mins", "min"}:
        return " minutes"
    if tokens & {"hours", "hour", "hrs"}:
        return " hours"
    if "mg" in tokens:
        return " mg"
    if "bpm" in tokens:
        return " bpm"
    if "percent" in tokens or "pct" in tokens:
        return "%"
    return ""


def _summary_for(column, series, concept=None):
    usable = series.dropna()
    kind = concept or ("Number" if pd.api.types.is_numeric_dtype(series) else "Category")
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(usable, errors="coerce").dropna()
        unit = _unit_for(column)
        main = f"Average {_number(values.mean())}{unit}"
        detail = (
            f"Typical range {_number(values.quantile(0.25))}{unit} to "
            f"{_number(values.quantile(0.75))}{unit}"
        )
    else:
        counts = usable.astype(str).value_counts()
        main = f"Most common: {counts.index[0]}"
        detail = f"{len(counts)} different response{'s' if len(counts) != 1 else ''}"
    return {"name": column, "kind": kind, "main": main, "detail": detail}


def _numeric_distribution(column, series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.nunique() <= 1:
        return None
    if values.nunique() <= 12:
        counts = values.value_counts().sort_index()
    else:
        counts = pd.cut(values, bins=7, duplicates="drop").value_counts().sort_index()
    return {
        "name": column,
        "type": "bar",
        "description": "Distribution across the available records",
        "labels": [str(label) for label in counts.index],
        "values": [int(value) for value in counts.values],
        "y_label": "Records",
    }


def _categorical_distribution(column, series):
    values = series.dropna().astype(str)
    counts = values.value_counts()
    if len(counts) <= 1 or len(counts) > 30:
        return None
    top = counts.head(8)
    labels = top.index.tolist()
    chart_values = [int(value) for value in top.values]
    if len(counts) > 8:
        labels.append("Other")
        chart_values.append(int(counts.iloc[8:].sum()))
    return {
        "name": column,
        "type": "bar",
        "description": "Most common responses",
        "labels": labels,
        "values": chart_values,
        "y_label": "Records",
    }


def _trend_chart(column, series, dates):
    values = pd.to_numeric(series, errors="coerce")
    dated = pd.DataFrame({"date": dates, "value": values}).dropna()
    if len(dated) < 4 or dated["date"].nunique() < 4:
        return None
    grouped = dated.groupby("date", sort=True)["value"].median()
    if grouped.nunique() <= 1:
        return None

    if len(grouped) > 20:
        chunk_size = math.ceil(len(grouped) / 20)
        reduced_labels = []
        reduced_values = []
        for start in range(0, len(grouped), chunk_size):
            chunk = grouped.iloc[start:start + chunk_size]
            reduced_labels.append(chunk.index[-1])
            reduced_values.append(float(chunk.median()))
        labels = reduced_labels
        values_out = reduced_values
    else:
        labels = grouped.index.tolist()
        values_out = [float(value) for value in grouped.values]

    return {
        "name": column,
        "type": "line",
        "description": "Typical value over time",
        "labels": [label.strftime("%d %b %y") for label in labels],
        "values": [round(value, 2) for value in values_out],
        "y_label": _display_name(column),
    }


def _friendly_label(column, concept):
    """Describe the selected measurement without exposing schema spelling."""
    name = _normalised(column)
    tokens = set(name.split("_"))
    if concept == "Sleep":
        if "latency" in name:
            return "Time to Fall Asleep"
        if any(word in name for word in ("quality", "rested")):
            return "Sleep Quality"
        if "wake" in name:
            return "Wake Time"
        if "bedtime" in name:
            return "Bedtime"
        return "Sleep Duration" if _unit_for(column) else "Sleep"
    if concept == "Activity":
        if "steps" in tokens:
            return "Daily Steps"
        if "distance" in tokens:
            return "Activity Distance"
        if "type" in tokens:
            return "Activity Type"
        return "Activity"
    if concept == "Physiology":
        if "hrv" in tokens:
            return "Heart Rate Variability"
        if tokens & {"spo2"}:
            return "Blood Oxygen"
        if "blood_pressure" in name or tokens & {"sbp", "dbp"}:
            return "Blood Pressure"
        return "Heart Rate"
    return {"Stress": "Stress Levels", "Screen Use": "Screen Time",
            "Mood / Well-Being": "Mood / Well-Being"}.get(concept, concept)


def build_dashboard_content(dataframe):
    """Return up to six wellness summaries and three data-based patterns."""
    date_match = _find_date_column(dataframe)
    date_column = date_match[0] if date_match else None
    dates = date_match[1] if date_match else None
    selected, _, _ = _select_columns(dataframe, date_column, limit=6)
    # Count all detected areas, independently of the six-card display limit.
    candidates = _detect_candidates(dataframe, date_column)
    areas = list(dict.fromkeys(item["concept"] for item in candidates))
    summaries = []
    charts = []
    for item in selected:
        column, concept = item["column"], item["concept"]
        series = dataframe[column]
        label = _friendly_label(column, concept)
        numeric = pd.api.types.is_numeric_dtype(series)
        summary = _summary_for(column, series, concept)
        summary["label"] = label
        summaries.append(summary)
        if len(charts) >= 3:
            continue
        chart = _trend_chart(column, series, dates) if numeric and dates is not None else None
        if chart is None:
            chart = _numeric_distribution(column, series) if numeric else _categorical_distribution(column, series)
        if chart is None:
            continue
        if chart["type"] == "line":
            description = (f"Typical recorded values range from {_number(min(chart['values']))}"
                           f" to {_number(max(chart['values']))}{_unit_for(column)} across the dates shown.")
        elif numeric:
            values = pd.to_numeric(series, errors="coerce").dropna()
            description = (f"The middle half of recorded values falls between {_number(values.quantile(.25))}"
                           f" and {_number(values.quantile(.75))}{_unit_for(column)}.")
        else:
            counts = series.dropna().astype(str).value_counts()
            description = f"{counts.index[0]} is the most common response ({int(counts.iloc[0]):,} of {int(counts.sum()):,} records with a value)."
        chart.update(title=label, description=description,
                     y_label=(label + _unit_for(column)) if chart["type"] == "line" else "Records")
        charts.append(chart)
    date_range = None
    if dates is not None:
        valid = dates.dropna()
        date_range = f"{valid.min():%d %b %Y} \u2013 {valid.max():%d %b %Y}"
    return {
        "column_summaries": summaries,
        "distribution_charts": charts,
        "wellness_detected": bool(areas),
        "wellness_column_count": len(areas),
        "wellness_areas": areas,
        "date_column": date_column,
        "date_range": date_range,
        "selection_note": None,
    }
