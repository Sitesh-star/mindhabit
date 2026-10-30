"""Rule-based, dataset-level habit pattern analysis for MindHabits."""

import math

import pandas as pd

from utils.wellbeing_detector import ID_HINTS, ID_NAMES


MINIMUM_VALUES = 6
MAXIMUM_INSIGHTS = 5

FAMILY_PRIORITY = {
    "sleep": 3,
    "activity": 3,
    "stress": 3,
    "screen": 2,
    "mindfulness": 2,
    "recovery": 2,
    "caffeine": 1,
    "alcohol": 1,
    "physiology": 1,
    "mental_state": 1,
}

HABIT_RULES = (
    {
        "key": "sleep_duration", "family": "sleep", "title": "Sleep Consistency",
        "icon": "bedtime", "polarity": "higher",
        "keywords": ("sleep_duration", "sleep_hours", "hours_slept", "slept_hours"),
        "positive_focus": "Keep maintaining regular sleep and wake times.",
        "improve_focus": "Consider keeping sleep and wake times more consistent.",
        "why": "Consistent sleep patterns may support steadier rest and recovery.",
    },
    {
        "key": "sleep_quality", "family": "sleep", "title": "Sleep Quality Pattern",
        "icon": "bedtime", "polarity": "higher",
        "keywords": ("sleep_quality", "quality_of_sleep", "sleep_efficiency"),
        "positive_focus": "Maintain the routines that appear to support your sleep quality.",
        "improve_focus": "Consider a more regular wind-down and sleep schedule.",
        "why": "A steadier sleep routine may support more consistent recovery.",
    },
    {
        "key": "sleep_latency", "family": "sleep", "title": "Settling to Sleep",
        "icon": "bedtime", "polarity": "lower",
        "keywords": ("sleep_latency", "time_to_sleep", "latency"),
        "positive_focus": "Keep maintaining a calm and consistent wind-down routine.",
        "improve_focus": "Try a calm, screen-free wind-down routine before bedtime.",
        "why": "A regular wind-down routine may make sleep timing more consistent.",
    },
    {
        "key": "wake_episodes", "family": "sleep", "title": "Night-time Waking",
        "icon": "bedtime", "polarity": "lower",
        "keywords": ("wake_episode", "night_waking", "awaken", "wake_after_sleep"),
        "positive_focus": "Maintain the sleep routine that supports this pattern.",
        "improve_focus": "Consider a quieter and more consistent bedtime routine.",
        "why": "Fewer interruptions may support more settled rest.",
    },
    {
        "key": "sleep_pattern", "family": "sleep", "title": "Sleep Pattern",
        "icon": "bedtime", "polarity": "higher",
        "keywords": ("sleep", "slept", "bedtime"),
        "positive_focus": "Keep maintaining regular sleep and wake times.",
        "improve_focus": "Consider making sleep and wake times more consistent.",
        "why": "A repeatable sleep routine may support steadier rest and recovery.",
    },
    {
        "key": "activity", "family": "activity", "title": "Activity Pattern",
        "icon": "directions_walk", "polarity": "higher",
        "keywords": ("steps", "activity", "exercise", "workout", "movement", "distance"),
        "positive_focus": "Keep maintaining regular movement throughout the week.",
        "improve_focus": "Consider adding manageable, regular movement to your routine.",
        "why": "Regular movement can support a balanced daily routine.",
    },
    {
        "key": "screen", "family": "screen", "title": "Screen Use Pattern",
        "icon": "smartphone", "polarity": "lower",
        "keywords": ("screen", "phone", "device_usage", "device_use"),
        "positive_focus": "Maintain the boundaries that keep screen use consistent.",
        "improve_focus": "Consider setting regular screen-free periods, especially before bedtime.",
        "why": "More consistent screen boundaries may support rest and attention.",
    },
    {
        "key": "caffeine", "family": "caffeine", "title": "Caffeine Pattern",
        "icon": "local_cafe", "polarity": "lower",
        "keywords": ("caffeine", "coffee"),
        "positive_focus": "Maintain consistent caffeine timing and keep it earlier in the day where possible.",
        "improve_focus": "Consider making caffeine intake more consistent and avoiding it late in the day.",
        "why": "Caffeine timing and consistency may influence rest routines.",
    },
    {
        "key": "alcohol", "family": "alcohol", "title": "Alcohol Pattern",
        "icon": "local_bar", "polarity": "lower",
        "keywords": ("alcohol",),
        "positive_focus": "Maintain a moderate and consistent pattern.",
        "improve_focus": "Consider limiting alcohol, particularly close to bedtime.",
        "why": "A more consistent pattern may better support sleep routines.",
    },
    {
        "key": "mindfulness", "family": "mindfulness", "title": "Mindfulness Practice",
        "icon": "self_improvement", "polarity": "higher",
        "keywords": ("mindfulness", "meditation", "relaxation"),
        "positive_focus": "Keep maintaining this regular mindfulness or relaxation practice.",
        "improve_focus": "Consider starting with short, regular mindfulness or relaxation sessions.",
        "why": "A repeatable relaxation practice may support day-to-day balance.",
    },
    {
        "key": "stress", "family": "stress", "title": "Stress Pattern",
        "icon": "spa", "polarity": "lower",
        "keywords": ("stress", "anxiety"),
        "positive_focus": "Maintain the routines that appear to support a steadier stress pattern.",
        "improve_focus": "Consider regular breaks, relaxation, or mindfulness during the day.",
        "why": "Recognising repeated stress patterns can help with planning restorative time.",
    },
    {
        "key": "hrv", "family": "recovery", "title": "Recovery Pattern",
        "icon": "monitor_heart", "polarity": "higher",
        "keywords": ("hrv", "heart_rate_variability"),
        "positive_focus": "Keep balancing activity with consistent sleep and recovery time.",
        "improve_focus": "Consider making rest, sleep, and recovery time more consistent.",
        "why": "A steadier recovery pattern may reflect a more balanced routine.",
    },
    {
        "key": "heart_rate", "family": "physiology", "title": "Heart Rate Pattern",
        "icon": "monitor_heart", "polarity": "neutral",
        "keywords": ("resting_heart_rate", "resting_hr", "heart_rate", "heartrate"),
        "positive_focus": "Continue balancing daily activity with enough recovery time.",
        "improve_focus": "Consider reviewing whether activity, rest, and recovery are consistent.",
        "why": "Repeated physiological patterns can help you reflect on routine consistency.",
    },
    {
        "key": "recovery", "family": "recovery", "title": "Recovery Pattern",
        "icon": "battery_charging_full", "polarity": "higher",
        "keywords": ("recovery", "rested", "refreshed", "fatigue", "tired", "rest"),
        "positive_focus": "Maintain the routines that appear to support recovery.",
        "improve_focus": "Consider allowing more consistent time for rest and recovery.",
        "why": "Repeated recovery patterns may show how well routines are being maintained.",
    },
    {
        "key": "wellbeing", "family": "mental_state", "title": "Emotional Well-Being Pattern",
        "icon": "sentiment_satisfied", "polarity": "higher",
        "keywords": ("mood", "emotion", "wellbeing", "well_being"),
        "positive_focus": "Keep maintaining the routines associated with the more positive records.",
        "improve_focus": "Consider noting which routines accompany more positive days.",
        "why": "Repeated emotional patterns can help connect daily routines with general well-being.",
    },
)


def _normalise_name(name):
    return str(name).strip().casefold().replace(" ", "_").replace("-", "_")


def _matches_keyword(column_name, keyword):
    name = _normalise_name(column_name)
    keyword = _normalise_name(keyword)
    if len(keyword) <= 4:
        return keyword in name.split("_")
    return keyword in name


def _is_identifier(column_name, series):
    name = _normalise_name(column_name)
    tokens = set(name.split("_"))
    if name in ID_NAMES or name.endswith("_id") or name.startswith("id_"):
        return True
    usable = series.dropna()
    if usable.empty:
        return False
    unique_ratio = usable.nunique(dropna=True) / len(usable)
    return unique_ratio >= 0.95 and bool(tokens & ID_HINTS)


def _find_date_column(dataframe):
    best = None
    for column in dataframe.columns:
        name = _normalise_name(column)
        tokens = set(name.split("_"))
        if "date" not in tokens and "timestamp" not in name:
            continue
        converted = pd.to_datetime(dataframe[column], errors="coerce", format="mixed")
        usable = int(converted.notna().sum())
        if usable >= MINIMUM_VALUES and (best is None or usable > best[2]):
            best = (column, converted, usable)
    return best


def _find_rule(column_name):
    name = _normalise_name(column_name)
    if "sleep_stage" in name:
        return None
    if name == "usage":
        return next(rule for rule in HABIT_RULES if rule["key"] == "screen")
    for rule in HABIT_RULES:
        if any(_matches_keyword(name, keyword) for keyword in rule["keywords"]):
            return rule
    return None


def _make_insight(rule, status, summary, score, pattern, trend=None):
    badges = {
        "positive": "Positive Pattern",
        "improve": "Suggested Focus",
        "neutral": "Neutral / Monitor",
    }
    return {
        "family": rule["family"],
        "title": rule["title"],
        "icon": rule["icon"],
        "status": status,
        "badge": badges[status],
        "summary": summary,
        "why": rule["why"],
        "focus": (
            rule["improve_focus"] if status == "improve"
            else rule["positive_focus"]
        ),
        "pattern": pattern,
        "trend": trend,
        "score": score,
    }


def _dated_trend(values, dates, rule):
    dated = pd.DataFrame({"date": dates, "value": values}).dropna()
    if len(dated) < 8 or dated["date"].nunique() < 4:
        return None

    by_date = dated.groupby("date", sort=True)["value"].median()
    segment_size = max(2, math.ceil(len(by_date) * 0.3))
    earlier = float(by_date.iloc[:segment_size].median())
    later = float(by_date.iloc[-segment_size:].median())
    overall = values.dropna()
    spread = float(overall.quantile(0.75) - overall.quantile(0.25))
    baseline = max(spread * 0.3, abs(float(overall.median())) * 0.06, 0.01)
    change = later - earlier
    if abs(change) < baseline or rule["polarity"] == "neutral":
        return None

    improving = change > 0 if rule["polarity"] == "higher" else change < 0
    status = "positive" if improving else "improve"
    direction = "a more supportive direction" if improving else "a less supportive direction"
    summary = (
        f"Later dated records show {rule['title'].casefold()} moving in {direction} "
        "compared with earlier records."
    )
    trend = "Improving" if improving else "Worsening"
    return _make_insight(rule, status, summary, 6, "Trend over time", trend)


def _non_directional_trend(values, dates, insight_status):
    """Label clearly stable or inconsistent dated patterns; otherwise return None."""
    dated = pd.DataFrame({"date": dates, "value": values}).dropna()
    if len(dated) < 8 or dated["date"].nunique() < 4:
        return None

    by_date = dated.groupby("date", sort=True)["value"].median()
    mean = float(by_date.mean())
    if not mean:
        return None
    variation = float(by_date.std() / abs(mean))
    if variation <= 0.12:
        return "Stable"
    if variation >= 0.35 and insight_status != "positive":
        return "Inconsistent"
    return None


def _numeric_pattern(series, dates, rule, column_name):
    values = pd.to_numeric(series, errors="coerce")
    usable = values.dropna()
    if len(usable) < MINIMUM_VALUES or usable.nunique() < 2:
        return None

    trend_rule = rule
    if rule["key"] == "recovery" and any(
        word in _normalise_name(column_name) for word in ("fatigue", "tired")
    ):
        trend_rule = {**rule, "polarity": "lower"}

    if dates is not None:
        trend = _dated_trend(values, dates, trend_rule)
        if trend:
            return trend

    median = float(usable.median())
    mean = float(usable.mean())
    q1 = float(usable.quantile(0.25))
    q3 = float(usable.quantile(0.75))
    variation = float(usable.std() / abs(mean)) if mean else float("inf")
    zero_share = float((usable == 0).mean())
    name = _normalise_name(column_name)

    if rule["key"] == "sleep_duration":
        appears_to_be_hours = any(unit in name for unit in ("hour", "_hr", "hrs"))
        if appears_to_be_hours and median < 6:
            return _make_insight(
                rule, "improve",
                "Sleep duration stays in a relatively short range across many records.",
                5, "Repeated lower sleep duration",
            )
        if variation >= 0.20:
            return _make_insight(
                rule, "improve",
                "Sleep duration is less stable across the uploaded records.",
                5, "Inconsistent pattern",
            )
        if variation <= 0.15:
            return _make_insight(
                rule, "positive",
                "Sleep duration remains fairly consistent across the uploaded records.",
                5, "Consistent pattern",
            )

    if rule["key"] == "sleep_quality":
        if variation >= 0.25:
            return _make_insight(
                rule, "improve", "Sleep quality changes noticeably between records.",
                4, "Inconsistent pattern",
            )
        if variation <= 0.15:
            return _make_insight(
                rule, "positive", "Sleep quality remains fairly consistent over the dataset.",
                3, "Consistent pattern",
            )

    if rule["key"] == "sleep_pattern":
        if variation >= 0.25:
            return _make_insight(
                rule, "improve", "Sleep-related values are inconsistent across the records.",
                4, "Inconsistent pattern",
            )
        if variation <= 0.15:
            return _make_insight(
                rule, "positive", "Sleep-related values remain fairly consistent across records.",
                3, "Consistent pattern",
            )

    if rule["key"] == "sleep_latency":
        appears_to_be_minutes = "min" in name or "minute" in name
        if appears_to_be_minutes and median >= 30:
            return _make_insight(
                rule, "improve", "Longer settling-to-sleep periods occur repeatedly in the dataset.",
                5, "Repeated longer latency",
            )
        if variation <= 0.30:
            return _make_insight(
                rule, "positive", "Time taken to settle to sleep is fairly consistent.",
                3, "Consistent pattern",
            )

    if rule["key"] == "wake_episodes":
        if q3 > max(median * 1.35, q1):
            return _make_insight(
                rule, "improve", "Some records show night-time waking well above the usual pattern.",
                4, "Higher-range periods",
            )
        if variation <= 0.25:
            return _make_insight(
                rule, "positive", "Night-time waking remains fairly consistent across records.",
                3, "Consistent pattern",
            )

    if rule["key"] == "activity":
        if zero_share >= 0.40:
            return _make_insight(
                rule, "improve", "Little or no activity is recorded on many occasions.",
                5, "Low activity frequency",
            )
        if variation > 0.70:
            return _make_insight(
                rule, "improve", "Activity levels are less stable between records.",
                4, "Inconsistent pattern",
            )
        if zero_share < 0.20 and variation <= 0.60:
            return _make_insight(
                rule, "positive", "Activity appears regularly across the uploaded records.",
                4, "Regular activity",
            )

    if rule["key"] == "mindfulness":
        if zero_share >= 0.40:
            return _make_insight(
                rule, "improve", "Little or no mindfulness time is recorded on many occasions.",
                5, "Low practice frequency",
            )
        if zero_share <= 0.25:
            return _make_insight(
                rule, "positive", "Mindfulness or relaxation appears regularly in the dataset.",
                4, "Regular practice",
            )

    if rule["key"] in {"screen", "caffeine", "alcohol"}:
        upper_range_jump = q3 > max(median * 1.45, q1)
        if variation >= 0.50 or upper_range_jump:
            summaries = {
                "screen": "Screen use is frequently higher and less stable across the dataset.",
                "caffeine": "Caffeine intake is inconsistent, with repeated higher-use records.",
                "alcohol": "Alcohol use is less stable, with frequent higher-range records.",
            }
            patterns = {
                "screen": "Frequently higher use",
                "caffeine": "Inconsistent intake",
                "alcohol": "Less stable pattern",
            }
            return _make_insight(
                rule, "improve", summaries[rule["key"]],
                4, patterns[rule["key"]],
            )
        if variation <= 0.25:
            summaries = {
                "screen": "Screen use remains relatively consistent across the records.",
                "caffeine": "Caffeine intake is usually within its normal dataset range.",
                "alcohol": "Alcohol use remains stable across the uploaded records.",
            }
            return _make_insight(
                rule, "neutral", summaries[rule["key"]],
                2, "Monitor this stable pattern",
            )

    if rule["key"] == "stress":
        if variation >= 0.30 or q3 > max(median * 1.35, q1):
            return _make_insight(
                rule, "improve", "Stress levels show repeated higher or less consistent periods.",
                5, "Higher or inconsistent periods",
            )
        if variation <= 0.20:
            return _make_insight(
                rule, "neutral", "Stress levels remain stable across the uploaded records.",
                2, "Stable pattern to monitor",
            )

    if rule["key"] in {"hrv", "recovery"}:
        inverse_measure = any(word in name for word in ("fatigue", "tired"))
        if variation >= 0.35:
            return _make_insight(
                rule, "improve", "Recovery-related values change noticeably across the dataset.",
                4, "Inconsistent recovery pattern",
            )
        if variation <= 0.22 and not inverse_measure:
            return _make_insight(
                rule, "positive", "Recovery-related values remain fairly consistent over time.",
                3, "Consistent recovery pattern",
            )
        return _make_insight(
            rule, "neutral", "Recovery-related values usually remain within their normal range.",
            2, "Monitor recovery consistency",
        )

    if rule["key"] == "heart_rate":
        if variation >= 0.25:
            return _make_insight(
                rule, "improve", "Heart rate values change noticeably across the uploaded records.",
                3, "Variable physiological pattern",
            )
        return _make_insight(
            rule, "neutral", "Heart rate values remain relatively consistent across the records.",
            2, "Stable pattern to monitor",
        )

    return None


def _categorical_pattern(series, rule):
    usable = series.dropna().astype(str).str.strip().str.casefold()
    if len(usable) < MINIMUM_VALUES or usable.nunique() < 2:
        return None

    if rule["key"] == "wellbeing":
        positive = {"good", "very_good", "great", "happy", "positive", "calm"}
        difficult = {"bad", "very_bad", "poor", "low", "sad", "anxious", "stressed"}
        positive_share = float(usable.isin(positive).mean())
        difficult_share = float(usable.isin(difficult).mean())
        if difficult_share >= 0.40:
            return _make_insight(
                rule, "improve", "Less positive emotional states appear repeatedly across the records.",
                2, "Recurring emotional pattern",
            )
        if positive_share >= 0.50:
            return _make_insight(
                rule, "positive", "More positive emotional states appear regularly across the records.",
                2, "Recurring positive pattern",
            )

    negative = {"none", "no", "false", "inactive", "never", "0", "rest"}
    participation = float((~usable.isin(negative)).mean())
    if rule["key"] in {"activity", "mindfulness"}:
        if participation <= 0.40:
            return _make_insight(
                rule, "improve", f"{rule['title']} is absent from many uploaded records.",
                4, "Low recorded frequency",
            )
        if participation >= 0.75:
            return _make_insight(
                rule, "positive", f"{rule['title']} appears regularly across the records.",
                3, "Regular recorded pattern",
            )
    return None


def _select_insights(candidates):
    grouped = {}
    for insight in candidates:
        grouped.setdefault(insight["family"], []).append(insight)

    best_by_family = {}
    for family, family_insights in grouped.items():
        family_insights.sort(key=lambda item: -item["score"])
        strongest = family_insights[0]
        matching_evidence = sum(
            item["status"] == strongest["status"] for item in family_insights
        )
        strongest["score"] += min(matching_evidence - 1, 2) * 0.5
        best_by_family[family] = strongest

    ordered = sorted(
        best_by_family.values(),
        key=lambda item: (
            -(item["score"] + FAMILY_PRIORITY.get(item["family"], 0)),
            item["status"] == "neutral",
            item["title"],
        ),
    )
    selected = ordered[:MAXIMUM_INSIGHTS]

    positive_candidates = [item for item in ordered if item["status"] == "positive"]
    if selected and positive_candidates and not any(
        item["status"] == "positive" for item in selected
    ):
        selected[-1] = positive_candidates[0]
    improvement_candidates = [item for item in ordered if item["status"] == "improve"]
    if selected and improvement_candidates and not any(
        item["status"] == "improve" for item in selected
    ):
        selected[-1] = improvement_candidates[0]
    return selected


def _presentation_for_insight(insight):
    """Return concise page wording without changing the underlying evidence."""
    family = insight["family"]
    status = insight["status"]
    title = {
        "sleep": "Sleep Routine",
        "activity": "Activity",
        "stress": "Stress",
        "screen": "Screen Time",
        "mindfulness": "Mindfulness",
        "recovery": "Recovery",
        "caffeine": "Caffeine",
        "alcohol": "Alcohol",
        "physiology": "Heart Rate",
        "mental_state": "Emotional Well-Being",
    }.get(family, insight["title"])

    trend_summary = {
        "Improving": f"Your {title.casefold()} appears to be improving.",
        "Worsening": f"Your {title.casefold()} may be getting worse.",
        "Stable": f"Your {title.casefold()} remains stable.",
        "Inconsistent": f"Your {title.casefold()} is inconsistent.",
    }
    summaries = {
        "positive": {
            "sleep": "Your sleep routine is relatively consistent.",
            "activity": "Your activity level is consistently strong.",
            "mindfulness": "Mindfulness appears regularly in your routine.",
            "recovery": "Your recovery pattern is relatively consistent.",
            "mental_state": "Positive emotional patterns appear regularly.",
        },
        "improve": {
            "sleep": "Your sleep routine is inconsistent or may need more attention.",
            "activity": "Your activity level is often low or inconsistent.",
            "stress": "Your stress level is often high or inconsistent.",
            "screen": "Your screen use is often high or inconsistent.",
            "mindfulness": "Mindfulness is missing from many records.",
            "recovery": "Your recovery pattern is inconsistent.",
            "caffeine": "Your caffeine intake is often high or inconsistent.",
            "alcohol": "Your alcohol intake is often high or inconsistent.",
            "physiology": "Your heart rate pattern is inconsistent.",
            "mental_state": "Less positive emotional patterns appear regularly.",
        },
        "neutral": {
            "stress": "Your stress pattern is stable, but worth watching.",
            "screen": "Your screen use is relatively consistent.",
            "caffeine": "Your caffeine intake is relatively stable.",
            "alcohol": "Your alcohol intake is relatively stable.",
            "recovery": "Your recovery pattern is stable, but worth watching.",
            "physiology": "Your heart rate pattern is relatively stable.",
        },
    }
    fallback = {
        "positive": "This habit appears consistent and supportive.",
        "improve": "This habit may need more attention.",
        "neutral": "This habit is stable, but worth keeping an eye on.",
    }
    summary = trend_summary.get(insight.get("trend"))
    if summary is None:
        summary = summaries.get(status, {}).get(family, fallback[status])

    action = insight["focus"]
    if status == "positive" and action.startswith("Keep maintaining"):
        action = "Continue" + action[len("Keep maintaining"):]

    return {
        "display_title": title,
        "display_status": {
            "positive": "Going well",
            "improve": "Needs attention",
            "neutral": "Keep an eye on",
        }[status],
        "display_summary": summary,
        "display_action": action,
    }


def analyze_habit_insights(dataframe):
    """Return up to five useful dataset-level habit patterns when available."""
    dataframe = dataframe.copy()
    date_match = _find_date_column(dataframe)
    date_column = date_match[0] if date_match else None
    dates = date_match[1] if date_match else None
    candidates = []
    detected_columns = []

    for column in dataframe.columns:
        if column == date_column or _is_identifier(column, dataframe[column]):
            continue
        rule = _find_rule(column)
        if rule is None:
            continue

        detected_columns.append(column)
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        numeric_share = float(numeric.notna().mean()) if len(dataframe) else 0
        if numeric_share >= 0.70:
            insight = _numeric_pattern(dataframe[column], dates, rule, column)
        else:
            insight = _categorical_pattern(dataframe[column], rule)
        if insight:
            if dates is not None and not insight.get("trend") and numeric_share >= 0.70:
                insight["trend"] = _non_directional_trend(
                    numeric, dates, insight["status"]
                )
            insight["source_column"] = column
            candidates.append(insight)

    insights = _select_insights(candidates)
    for insight in insights:
        insight.update(_presentation_for_insight(insight))
        insight.pop("score", None)
        insight.pop("family", None)
        insight.pop("source_column", None)

    positive_count = sum(item["status"] == "positive" for item in insights)
    improvement_count = sum(item["status"] == "improve" for item in insights)
    neutral_count = sum(item["status"] == "neutral" for item in insights)
    return {
        "insights": insights,
        "positive_count": positive_count,
        "improvement_count": improvement_count,
        "neutral_count": neutral_count,
        "total_insights": len(insights),
        "detected_column_count": len(set(detected_columns)),
        "records_analyzed": len(dataframe),
        "date_column": date_column,
        "date_display_name": (
            _normalise_name(date_column).replace("_", " ").title()
            if date_column else None
        ),
    }
