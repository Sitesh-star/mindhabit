"""Professional, user-facing PDF report generation for MindHabits.

The renderer is deliberately self-contained: it draws text, cards, and charts as
PDF vector commands and does not retrain models or require a PDF/chart package.
"""

import math
import textwrap
import unicodedata
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from utils.dashboard_analyzer import build_dashboard_content, prepare_dashboard_dataframe


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 44
CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2

TEAL = "2F7468"
TEAL_DARK = "205A51"
TEAL_LIGHT = "E7F4F1"
AMBER = "B7791F"
AMBER_LIGHT = "FFF4DE"
INK = "172027"
MUTED = "647078"
BORDER = "D9E2E0"
SOFT = "F5F8F7"
WHITE = "FFFFFF"

DISCLAIMER = (
    "MindHabits provides general wellness and habit insights for awareness and "
    "self-reflection. The results are not a medical diagnosis and should not "
    "replace advice from a qualified healthcare professional."
)

CONCEPT_RULES = OrderedDict([
    ("sleep", ("sleep", "slept", "bedtime", "latency", "wake", "rested")),
    ("stress", ("stress", "anxiety", "tension")),
    ("mood", ("mood", "emotion", "wellbeing", "well_being", "happiness")),
    ("activity", ("steps", "activity", "exercise", "workout", "movement", "distance")),
    ("screen", ("screen", "phone", "device_use", "digital", "usage")),
    ("mindfulness", ("mindfulness", "meditation", "relaxation")),
    ("caffeine", ("caffeine", "coffee")),
    ("alcohol", ("alcohol",)),
    ("recovery", ("recovery", "hrv", "heart_rate_variability", "fatigue", "tired")),
    ("heart_rate", ("resting_heart_rate", "heart_rate", "heartrate", "pulse")),
])

CONCEPT_PRIORITY = {
    "sleep": 10, "stress": 9, "mood": 9, "activity": 8,
    "screen": 7, "recovery": 7, "mindfulness": 6,
    "caffeine": 5, "alcohol": 4, "heart_rate": 4,
}

MEANINGS = {
    "sleep": "Sleep patterns may be associated with recovery, energy, and daily well-being.",
    "stress": "Repeated stress patterns may influence rest, focus, and day-to-day balance.",
    "mood": "Mood patterns may appear alongside changes in routines and daily well-being.",
    "activity": "Regular movement may support energy, sleep, and stress management.",
    "screen": "Screen-use patterns may influence attention and the consistency of evening routines.",
    "mindfulness": "Regular relaxation or mindfulness may support stress management and recovery.",
    "caffeine": "Caffeine amount and timing may influence energy and sleep routines.",
    "alcohol": "Alcohol patterns may be associated with sleep and recovery routines.",
    "recovery": "Recovery measures may reflect how activity, rest, and sleep patterns work together.",
    "heart_rate": "Heart-rate patterns may be useful when reflecting on activity and recovery routines.",
}

RECOMMENDATION_CATALOG = {
    "sleep": {
        "improve": "Improve Your Sleep Routine", "positive": "Maintain Your Sleep Routine",
        "help": "More consistent sleep and wake patterns may support recovery, energy, and daily well-being.",
        "actions": [
            "Aim to sleep and wake within a similar time range each day.",
            "Keep large weekday and weekend sleep changes to a minimum.",
            "Use a short, calm wind-down routine before bed.",
        ],
    },
    "stress": {
        "improve": "Build Regular Stress-Management Breaks", "positive": "Maintain Your Stress-Supporting Routine",
        "help": "Short restorative breaks may support focus, relaxation, and a steadier daily routine.",
        "actions": [
            "Schedule one or two brief pauses during demanding parts of the day.",
            "Try a short breathing, stretching, or quiet relaxation exercise.",
            "Record stress consistently to see which routines are associated with calmer days.",
        ],
    },
    "mood": {
        "improve": "Support a More Consistent Well-Being Routine", "positive": "Maintain Your Positive Well-Being Routine",
        "help": "Regular restorative routines may support steadier day-to-day well-being.",
        "actions": [
            "Note which sleep, activity, and social routines accompany better days.",
            "Make time for one manageable restorative activity each day.",
            "Continue recording patterns so changes can be reviewed over time.",
        ],
    },
    "activity": {
        "improve": "Add More Regular Movement", "positive": "Maintain Your Activity Routine",
        "help": "Regular movement may support energy, sleep, stress management, and general well-being.",
        "actions": [
            "Choose a manageable daily movement goal that fits your routine.",
            "Break up long inactive periods with short walks or stretching.",
            "Keep activity consistent across the week rather than relying on one busy day.",
        ],
    },
    "screen": {
        "improve": "Reduce Unnecessary Screen Use", "positive": "Maintain Healthy Screen Boundaries",
        "help": "More consistent screen boundaries may support attention and a calmer evening routine.",
        "actions": [
            "Reduce unnecessary device use during the last 30-60 minutes before bed.",
            "Replace some evening screen time with a quieter activity.",
            "Monitor whether lower evening use appears alongside better sleep records.",
        ],
    },
    "mindfulness": {
        "improve": "Build a Short Mindfulness Routine", "positive": "Maintain Your Mindfulness Routine",
        "help": "A repeatable relaxation practice may support stress management and daily balance.",
        "actions": [
            "Start with a short session at the same time each day.",
            "Use breathing, quiet reflection, or gentle stretching as a simple option.",
            "Increase the time gradually only if the routine feels sustainable.",
        ],
    },
    "caffeine": {
        "improve": "Review Caffeine Amount and Timing", "positive": "Maintain Moderate Caffeine Habits",
        "help": "Keeping caffeine moderate and earlier in the day may support a more consistent sleep routine.",
        "actions": [
            "Keep caffeine earlier in the day where possible.",
            "Reduce large changes in intake between days.",
            "Track whether timing changes appear alongside better sleep or recovery records.",
        ],
    },
    "alcohol": {
        "improve": "Review Alcohol Timing and Consistency", "positive": "Maintain a Moderate Alcohol Pattern",
        "help": "A moderate, consistent pattern may better support sleep and recovery routines.",
        "actions": [
            "Avoid placing alcohol close to bedtime where possible.",
            "Keep intake moderate and avoid large changes between days.",
            "Compare future sleep records with days when alcohol use is lower.",
        ],
    },
    "recovery": {
        "improve": "Give Recovery More Consistent Attention", "positive": "Maintain Your Recovery Routine",
        "help": "Balancing activity with sleep and recovery time may support steadier energy and well-being.",
        "actions": [
            "Plan lighter periods after demanding days.",
            "Keep sleep and activity records consistent so recovery patterns are easier to review.",
            "Use gentle movement or relaxation when additional recovery is needed.",
        ],
    },
    "heart_rate": {
        "improve": "Support a More Consistent Recovery Routine", "positive": "Continue Monitoring Recovery",
        "help": "Consistent rest and activity routines may support more stable physiological patterns.",
        "actions": [
            "Balance active periods with enough rest.",
            "Record measurements under similar conditions when possible.",
            "Review longer-term patterns instead of reacting to one value.",
        ],
    },
}


def _plain(value):
    normalised = unicodedata.normalize("NFKD", str(value))
    return normalised.encode("ascii", "replace").decode("ascii")


def _escape(value):
    return _plain(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _normalise(value):
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _keyword_match(value, keyword):
    name = _normalise(value)
    keyword = _normalise(keyword)
    return keyword in name.split("_") if len(keyword) <= 4 else keyword in name


def _concept_for(*values):
    joined = "_".join(_normalise(value) for value in values if value)
    for concept, keywords in CONCEPT_RULES.items():
        if any(_keyword_match(joined, keyword) for keyword in keywords):
            return concept
    return None


def _friendly_name(value):
    normalised = _normalise(value)
    if "sleep" in normalised and any(token in normalised for token in ("duration", "hour", "slept")):
        return "Sleep Duration"
    if any(token in normalised for token in ("screen", "phone")):
        return "Screen Time"
    if "stress" in normalised:
        return "Stress Level"
    if "steps" in normalised:
        return "Daily Activity"
    if "hrv" in normalised or "heart_rate_variability" in normalised:
        return "HRV"
    if "heart_rate" in normalised or "heartrate" in normalised:
        return "Heart Rate"
    tokens = [
        token for token in normalised.split("_")
        if token not in {"min", "mins", "minute", "minutes", "hr", "hrs", "hour", "hours", "mg", "bpm"}
    ]
    replacements = {"hrv": "HRV", "spo2": "SpO2", "hr": "Heart Rate"}
    return " ".join(replacements.get(token, token.title()) for token in tokens) or "Wellness Factor"


def _unit_for(column):
    tokens = set(_normalise(column).split("_"))
    if tokens & {"min", "mins", "minute", "minutes"}:
        return " minutes"
    if tokens & {"hr", "hrs", "hour", "hours"}:
        return " hours"
    if "mg" in tokens:
        return " mg"
    if "bpm" in tokens:
        return " bpm"
    if tokens & {"percent", "percentage", "pct"}:
        return "%"
    return ""


def _number(value):
    rounded = round(float(value), 2)
    return str(int(rounded)) if rounded.is_integer() else str(rounded)


def _chart_label(value):
    label = str(value).strip()
    if len(label) > 2 and label[0] in "([" and label[-1] in ")]" and "," in label:
        lower, upper = label[1:-1].split(",", 1)
        return f"{lower.strip()} to {upper.strip()}"
    return label


def _generated_label(generated_at=None):
    generated_at = generated_at or datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))
    return generated_at.strftime("%d %b %Y, %I:%M %p %Z").strip()


def _prepare_data(dataframe):
    if dataframe is None:
        return None, {"records": None, "date_range": None, "charts": []}
    frame = prepare_dashboard_dataframe(dataframe)
    dashboard = build_dashboard_content(frame)
    date_range = None
    date_column = dashboard.get("date_column")
    if date_column and date_column in frame:
        dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed").dropna()
        if not dates.empty:
            date_range = f"{dates.min().strftime('%d %b %Y')} to {dates.max().strftime('%d %b %Y')}"
    charts = []
    for chart in dashboard.get("distribution_charts", [])[:4]:
        chart = dict(chart)
        chart["labels"] = [_chart_label(label) for label in chart.get("labels", [])]
        chart["display_name"] = _friendly_name(chart.get("name", ""))
        chart["concept"] = _concept_for(chart.get("name"))
        if chart["concept"] is None:
            continue
        chart["title"] = (
            f"{chart['display_name']} Over Time"
            if chart.get("type") == "line"
            else f"{chart['display_name']} Distribution"
        )
        chart["explanation"] = _chart_explanation(chart)
        charts.append(chart)
    return frame, {"records": len(frame), "date_range": date_range, "charts": charts}


def _chart_explanation(chart):
    values = [float(value) for value in chart.get("values", [])]
    name = chart.get("display_name", "This measure").casefold()
    concept = chart.get("concept")
    if not values:
        first = f"The available {name} records are shown above."
    elif chart.get("type") == "line" and len(values) >= 2:
        span = max(values) - min(values)
        change = values[-1] - values[0]
        if span == 0 or abs(change) <= span * 0.15:
            first = f"Recorded {name} values remain relatively stable across the recorded period."
        elif change > 0:
            first = f"Recorded {name} values generally increase across the recorded period."
        else:
            first = f"Recorded {name} values generally decrease across the recorded period."
    else:
        peak = max(range(len(values)), key=values.__getitem__)
        labels = chart.get("labels", [])
        label = labels[peak] if peak < len(labels) else "one range"
        first = f"The most common {name} group in these records is {label}."
    meaning = MEANINGS.get(concept)
    return f"{first} {meaning}" if meaning else first


def _factor_details(frame, wellbeing_result):
    details = []
    seen = set()
    for factor in (wellbeing_result or {}).get("contributing_factors", [])[:5]:
        name = factor.get("name") or _friendly_name(factor.get("column", ""))
        concept = _concept_for(factor.get("column"), name)
        dedupe_key = concept or _normalise(name)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        column = factor.get("column")
        pattern = "This factor was meaningfully associated with the analysis result."
        if frame is not None and column in frame.columns:
            series = frame[column].dropna()
            if not series.empty and pd.api.types.is_numeric_dtype(frame[column]):
                values = pd.to_numeric(series, errors="coerce").dropna()
                latest = float(values.iloc[-1])
                q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
                unit = _unit_for(column)
                if latest < q1:
                    position_label = "lower than the typical range"
                elif latest > q3:
                    position_label = "higher than the typical range"
                else:
                    position_label = "within the typical range"
                pattern = f"The latest recorded value is {_number(latest)}{unit}, which is {position_label} in your dataset."
            elif not series.empty:
                latest = str(series.iloc[-1]).strip()
                common = str(series.astype(str).mode().iloc[0]).strip()
                pattern = f"The latest recorded value is {latest}. The most common response in your data is {common}."
        details.append({
            "name": name,
            "concept": concept,
            "pattern": pattern,
            "position": (
                "low" if "lower than" in pattern.casefold()
                else "high" if "higher than" in pattern.casefold()
                else "typical" if "within the typical" in pattern.casefold()
                else "unknown"
            ),
            "meaning": MEANINGS.get(concept, "This pattern appears related to the overall analysis, but it does not prove cause and effect."),
        })
    return details


def _habit_items(habit_result):
    items = []
    status_labels = {"positive": "Going Well", "improve": "Needs Attention", "neutral": "Keep an Eye On"}
    for insight in (habit_result or {}).get("insights", [])[:5]:
        title = insight.get("display_title") or insight.get("title", "Habit")
        trend = insight.get("trend")
        if trend == "Worsening":
            trend = "Getting worse"
        items.append({
            "title": title,
            "status": insight.get("status", "neutral"),
            "status_label": status_labels.get(insight.get("status"), "Keep an Eye On"),
            "summary": insight.get("display_summary") or insight.get("summary", ""),
            "why": insight.get("why") or MEANINGS.get(_concept_for(title), "This recurring pattern may be useful when reviewing your daily routine."),
            "action": insight.get("display_action") or insight.get("focus", "Continue monitoring this pattern."),
            "trend": trend,
            "concept": _concept_for(title, insight.get("title")),
        })
    return items


def _expanded_recommendations(factors, habits, wellbeing_result, frame):
    candidates = {}

    def add(concept, status, evidence, score, source):
        if concept not in RECOMMENDATION_CATALOG:
            return
        item = candidates.setdefault(concept, {
            "concept": concept, "status": status, "evidence": [], "score": 0, "sources": set(),
        })
        if status == "improve" or (status == "positive" and item["status"] == "neutral"):
            item["status"] = status
        if evidence and evidence not in item["evidence"]:
            item["evidence"].append(evidence)
        item["score"] = max(item["score"], score)
        item["sources"].add(source)

    for factor in factors:
        status = "neutral"
        position = factor.get("position")
        if factor["concept"] in {"activity", "mindfulness", "recovery"}:
            status = "positive" if position == "high" else "improve" if position == "low" else "neutral"
        elif factor["concept"] in {"stress", "screen", "caffeine", "alcohol", "heart_rate"}:
            status = "positive" if position == "low" else "improve" if position == "high" else "neutral"
        elif factor["concept"] == "sleep" and position in {"low", "high"}:
            status = "improve"
        add(factor["concept"], status, factor["pattern"], 6, "well-being result")

    for habit in habits:
        score = {"improve": 10, "positive": 7, "neutral": 4}.get(habit["status"], 4)
        add(habit["concept"], habit["status"], habit["summary"], score, "habit patterns")

    for action in (wellbeing_result or {}).get("recommendations", []):
        concept = _concept_for(action)
        add(concept, "neutral", action, 3, "well-being result")

    if frame is not None:
        for concept, item in candidates.items():
            matching = [column for column in frame if _concept_for(column) == concept]
            if matching:
                completeness = max(float(frame[column].notna().mean()) for column in matching)
                item["score"] += completeness * 2

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -(item["score"] + CONCEPT_PRIORITY.get(item["concept"], 0) + len(item["sources"])),
            item["concept"],
        ),
    )[:5]
    recommendations = []
    for item in ranked:
        catalog = RECOMMENDATION_CATALOG[item["concept"]]
        status_key = "positive" if item["status"] == "positive" else "improve"
        evidence = " ".join(item["evidence"][:2]) or "A recurring pattern in your data supports this focus."
        recommendations.append({
            "concept": item["concept"],
            "title": catalog[status_key],
            "why_suggested": evidence,
            "why_help": catalog["help"],
            "actions": catalog["actions"],
            "positive": item["status"] == "positive",
        })
    return recommendations


def build_report_content(dataset_name, wellbeing_result, habit_result, generated_at=None, dataframe=None):
    """Build report-ready content without rerunning any analysis model."""
    frame, overview = _prepare_data(dataframe)
    factors = _factor_details(frame, wellbeing_result)
    habits = _habit_items(habit_result)
    recommendations = _expanded_recommendations(factors, habits, wellbeing_result, frame)
    prediction = (wellbeing_result or {}).get("prediction", {})
    confidence_band = prediction.get("confidence_band") or ""
    confidence_label = confidence_band.replace(" confidence", "").strip().title()
    return {
        "dataset_name": dataset_name or "Not available",
        "generated": _generated_label(generated_at),
        "records": overview["records"],
        "date_range": overview["date_range"],
        "wellbeing": wellbeing_result,
        "prediction": prediction,
        "confidence_label": confidence_label,
        "factors": factors,
        "charts": overview["charts"],
        "habit_result": habit_result,
        "habits": habits,
        "recommendations": recommendations,
        "next_steps": [item["title"] for item in recommendations[:3]],
    }


class _PDF:
    def __init__(self):
        self.pages = []
        self.commands = None

    @staticmethod
    def _rgb(hex_color):
        return tuple(int(hex_color[index:index + 2], 16) / 255 for index in (0, 2, 4))

    def new_page(self, section=None):
        self.commands = []
        self.pages.append(self.commands)
        if section:
            self.text(MARGIN, 808, "MINDHABITS", 8, True, TEAL)
            self.text(PAGE_WIDTH - MARGIN, 808, section.upper(), 8, True, MUTED, align="right")
            self.line(MARGIN, 798, PAGE_WIDTH - MARGIN, 798, BORDER, 0.7)
        return 770

    def text(self, x, y, value, size=10, bold=False, color=INK, align="left"):
        r, g, b = self._rgb(color)
        safe = _escape(value)
        estimated = len(_plain(value)) * size * 0.5
        if align == "right":
            x -= estimated
        elif align == "center":
            x -= estimated / 2
        font = "F2" if bold else "F1"
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg BT /{font} {size} Tf {x:.1f} {y:.1f} Td ({safe}) Tj ET")

    def wrapped(self, x, y, value, width, size=10, bold=False, color=INK, leading=None, max_lines=None):
        leading = leading or size + 4
        chars = max(16, int(width / (size * 0.52)))
        lines = textwrap.wrap(_plain(value), width=chars, break_long_words=False, break_on_hyphens=False) or [""]
        if max_lines:
            lines = lines[:max_lines]
        for line_text in lines:
            self.text(x, y, line_text, size, bold, color)
            y -= leading
        return y

    def rect(self, x, y, width, height, fill=WHITE, stroke=BORDER, line_width=0.8):
        fr, fg, fb = self._rgb(fill)
        sr, sg, sb = self._rgb(stroke)
        self.commands.append(
            f"q {fr:.3f} {fg:.3f} {fb:.3f} rg {sr:.3f} {sg:.3f} {sb:.3f} RG "
            f"{line_width:.2f} w {x:.1f} {y:.1f} {width:.1f} {height:.1f} re B Q"
        )

    def line(self, x1, y1, x2, y2, color=BORDER, width=0.8):
        r, g, b = self._rgb(color)
        self.commands.append(f"q {r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S Q")

    def polyline(self, points, color=TEAL, width=2):
        if len(points) < 2:
            return
        r, g, b = self._rgb(color)
        path = [f"{points[0][0]:.1f} {points[0][1]:.1f} m"]
        path.extend(f"{x:.1f} {y:.1f} l" for x, y in points[1:])
        self.commands.append(f"q {r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {' '.join(path)} S Q")

    def build(self):
        page_count = len(self.pages)
        first_page_object = 5
        page_objects = [first_page_object + index * 2 for index in range(page_count)]
        objects = {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: (f"<< /Type /Pages /Count {page_count} /Kids [{' '.join(f'{number} 0 R' for number in page_objects)}] >>").encode("ascii"),
            3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        }
        for index, commands in enumerate(self.pages):
            self.commands = commands
            self.line(MARGIN, 42, PAGE_WIDTH - MARGIN, 42, BORDER, 0.6)
            self.text(MARGIN, 27, "MindHabits Wellness & Habit Report", 7, False, MUTED)
            self.text(PAGE_WIDTH - MARGIN, 27, f"Page {index + 1} of {page_count}", 7, False, MUTED, align="right")
            page_object = first_page_object + index * 2
            content_object = page_object + 1
            stream = "\n".join(commands).encode("ascii")
            objects[page_object] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_object} 0 R >>"
            ).encode("ascii")
            objects[content_object] = f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"

        output = bytearray(b"%PDF-1.4\n%MindHabits\n")
        offsets = {0: 0}
        for number in range(1, max(objects) + 1):
            offsets[number] = len(output)
            output.extend(f"{number} 0 obj\n".encode("ascii"))
            output.extend(objects[number])
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {max(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for number in range(1, max(objects) + 1):
            output.extend(f"{offsets[number]:010d} 00000 n \n".encode("ascii"))
        output.extend((f"trailer\n<< /Size {max(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF").encode("ascii"))
        return bytes(output)


def _section_title(pdf, y, title, subtitle=None):
    pdf.text(MARGIN, y, title, 19, True, TEAL_DARK)
    y -= 22
    if subtitle:
        y = pdf.wrapped(MARGIN, y, subtitle, CONTENT_WIDTH, 9, False, MUTED, 13)
    return y - 10


def _draw_overview_page(pdf, content):
    y = pdf.new_page()
    pdf.text(MARGIN, 800, "MINDHABITS", 9, True, TEAL)
    pdf.text(MARGIN, 758, "MindHabits Wellness & Habit Report", 25, True, INK)
    y = pdf.wrapped(
        MARGIN, 728,
        "This report summarises the main wellness patterns found in your data and provides practical suggestions that may help support healthier daily habits.",
        CONTENT_WIDTH, 11, False, MUTED, 16,
    ) - 10
    metadata = [
        ("Dataset", content["dataset_name"]),
        ("Generated", content["generated"]),
        ("Records analysed", str(content["records"]) if content["records"] is not None else "Not available"),
        ("Date range", content["date_range"] or "Not available"),
    ]
    card_width = (CONTENT_WIDTH - 12) / 2
    for index, (label, value) in enumerate(metadata):
        column, row = index % 2, index // 2
        x = MARGIN + column * (card_width + 12)
        top = y - row * 67
        pdf.rect(x, top - 54, card_width, 54, SOFT, BORDER)
        pdf.text(x + 12, top - 17, label.upper(), 7, True, MUTED)
        pdf.wrapped(x + 12, top - 34, value, card_width - 24, 10, True, INK, 12, 2)
    y -= 145

    wellbeing = content["wellbeing"]
    if wellbeing:
        prediction = content["prediction"]
        outcome = wellbeing.get("outcome_display_name", "Well-Being Outcome")
        value = prediction.get("value", "Not available")
        confidence = prediction.get("confidence")
        pdf.rect(MARGIN, y - 210, CONTENT_WIDTH, 210, TEAL_LIGHT, "BFDCD6", 1.2)
        pdf.text(MARGIN + 18, y - 27, "YOUR WELL-BEING RESULT", 9, True, TEAL)
        pdf.text(MARGIN + 18, y - 58, outcome, 18, True, INK)
        pdf.text(MARGIN + 18, y - 100, str(value), 30, True, TEAL_DARK)
        if confidence is not None:
            label = content["confidence_label"] or "Available"
            pdf.text(MARGIN + 18, y - 128, f"Confidence: {confidence}% - {label}", 11, True, INK)
        if wellbeing.get("problem_type") == "Regression":
            explanation = f"Based on your recent data, the estimated {outcome.casefold()} is {value}."
        else:
            explanation = f"For {outcome.casefold()}, the pattern in your recent data is closest to {value}."
        if prediction.get("confidence_band") == "Low confidence":
            explanation += " The confidence is low, so view this as an indication rather than a definite conclusion."
        pdf.wrapped(MARGIN + 18, y - 154, explanation, CONTENT_WIDTH - 36, 10, False, MUTED, 14, 4)


def _draw_factor_page(pdf, content):
    if not content["factors"]:
        return
    y = pdf.new_page("Key Wellness Factors")
    y = _section_title(pdf, y, "Key Wellness Factors", "Meaningful factors associated with the analysis result. These relationships do not prove cause and effect.")
    for factor in content["factors"]:
        height = 112
        if y - height < 62:
            y = pdf.new_page("Key Wellness Factors")
        pdf.rect(MARGIN, y - height, CONTENT_WIDTH, height, WHITE, BORDER)
        pdf.text(MARGIN + 14, y - 22, factor["name"], 13, True, TEAL_DARK)
        pdf.text(MARGIN + 14, y - 43, "CURRENT PATTERN / VALUE", 7, True, MUTED)
        next_y = pdf.wrapped(MARGIN + 14, y - 58, factor["pattern"], CONTENT_WIDTH - 28, 9, False, INK, 12, 2)
        pdf.text(MARGIN + 14, next_y - 4, "WHAT IT MAY MEAN", 7, True, MUTED)
        pdf.wrapped(MARGIN + 14, next_y - 19, factor["meaning"], CONTENT_WIDTH - 28, 9, False, MUTED, 12, 2)
        y -= height + 10


def _draw_chart(pdf, y, chart):
    height = 310
    pdf.rect(MARGIN, y - height, CONTENT_WIDTH, height, WHITE, BORDER)
    pdf.text(MARGIN + 14, y - 24, chart["title"], 13, True, TEAL_DARK)
    plot_x, plot_y = MARGIN + 48, y - 220
    plot_width, plot_height = CONTENT_WIDTH - 70, 155
    pdf.line(plot_x, plot_y, plot_x, plot_y + plot_height, MUTED, 0.7)
    pdf.line(plot_x, plot_y, plot_x + plot_width, plot_y, MUTED, 0.7)
    values = [float(value) for value in chart.get("values", [])]
    labels = [str(label) for label in chart.get("labels", [])]
    if values:
        low, high = min(values), max(values)
        if chart.get("type") == "line":
            padding = (high - low) * 0.12 or max(abs(high) * 0.1, 1)
            low, high = low - padding, high + padding
            points = []
            for index, value in enumerate(values):
                x = plot_x + (index / max(1, len(values) - 1)) * plot_width
                py = plot_y + ((value - low) / max(high - low, 1e-9)) * plot_height
                points.append((x, py))
            pdf.polyline(points, TEAL, 2)
            pdf.text(plot_x - 6, plot_y + plot_height - 2, _number(high), 6, False, MUTED, "right")
            pdf.text(plot_x - 6, plot_y + 1, _number(low), 6, False, MUTED, "right")
            pdf.text(plot_x, plot_y + plot_height + 9, chart["display_name"], 7, True, MUTED)
            pdf.text(plot_x + plot_width, plot_y - 27, "Date", 7, True, MUTED, "right")
            label_step = max(1, math.ceil(len(labels) / 5))
            for index in range(0, len(labels), label_step):
                x = plot_x + (index / max(1, len(labels) - 1)) * plot_width
                pdf.text(x, plot_y - 14, labels[index][:12], 6, False, MUTED, "center")
        else:
            maximum = max(high, 1)
            pdf.text(plot_x - 6, plot_y + plot_height - 2, _number(maximum), 6, False, MUTED, "right")
            pdf.text(plot_x, plot_y + plot_height + 9, "Records", 7, True, MUTED)
            pdf.text(plot_x + plot_width, plot_y - 27, "Value / Category", 7, True, MUTED, "right")
            gap = 5
            bar_width = max(8, (plot_width - gap * len(values)) / len(values))
            for index, value in enumerate(values):
                bar_height = (value / maximum) * (plot_height - 8)
                x = plot_x + index * (bar_width + gap) + gap / 2
                pdf.rect(x, plot_y, bar_width, bar_height, TEAL_LIGHT, TEAL, 0.5)
                pdf.text(x + bar_width / 2, plot_y - 14, labels[index][:10], 6, False, MUTED, "center")
    pdf.wrapped(MARGIN + 14, y - 246, chart["explanation"], CONTENT_WIDTH - 28, 9, False, MUTED, 12, 4)
    return y - height - 12


def _draw_graph_pages(pdf, content):
    charts = content["charts"]
    if not charts:
        return
    y = pdf.new_page("Important Graphs")
    y = _section_title(pdf, y, "Important Graphs", "Only the most useful wellness-related graphs from the current dataset are included.")
    for chart in charts:
        if y - 310 < 58:
            y = pdf.new_page("Important Graphs")
            y = _section_title(pdf, y, "Important Graphs")
        y = _draw_chart(pdf, y, chart)


def _draw_habits_page(pdf, content):
    if not content["habits"]:
        return
    y = pdf.new_page("Habit Insights")
    y = _section_title(pdf, y, "Your Habit Overview", "Broader recurring patterns found across the full dataset.")
    result = content["habit_result"] or {}
    counts = [
        ("Going well", result.get("positive_count", 0), TEAL_LIGHT, TEAL_DARK),
        ("Need attention", result.get("improvement_count", 0), AMBER_LIGHT, AMBER),
        ("Keep an eye on", result.get("neutral_count", 0), SOFT, MUTED),
    ]
    width = (CONTENT_WIDTH - 16) / 3
    for index, (label, count, fill, color) in enumerate(counts):
        x = MARGIN + index * (width + 8)
        pdf.rect(x, y - 65, width, 65, fill, BORDER)
        pdf.text(x + 10, y - 23, str(count), 20, True, color)
        pdf.text(x + 10, y - 44, label, 8, True, color)
    y -= 88
    pdf.text(MARGIN, y, "Habit Insights", 17, True, TEAL_DARK)
    y -= 20
    for habit in content["habits"]:
        height = 119
        if y - height < 58:
            y = pdf.new_page("Habit Insights")
        fill = TEAL_LIGHT if habit["status"] == "positive" else AMBER_LIGHT if habit["status"] == "improve" else SOFT
        accent = TEAL_DARK if habit["status"] == "positive" else AMBER if habit["status"] == "improve" else MUTED
        pdf.rect(MARGIN, y - height, CONTENT_WIDTH, height, WHITE, BORDER)
        pdf.rect(MARGIN, y - height, 5, height, accent, accent, 0)
        pdf.text(MARGIN + 15, y - 21, habit["title"], 12, True, INK)
        pdf.rect(PAGE_WIDTH - MARGIN - 104, y - 31, 92, 20, fill, fill, 0)
        pdf.text(PAGE_WIDTH - MARGIN - 96, y - 25, habit["status_label"], 7, True, accent)
        if habit["trend"]:
            pdf.text(MARGIN + 15, y - 40, f"Trend: {habit['trend']}", 7, True, accent)
        next_y = pdf.wrapped(MARGIN + 15, y - 57, habit["summary"], CONTENT_WIDTH - 30, 9, False, INK, 12, 2)
        pdf.wrapped(MARGIN + 15, next_y - 3, f"Why this matters: {habit['why']}", CONTENT_WIDTH - 30, 8, False, MUTED, 11, 2)
        pdf.wrapped(MARGIN + 15, y - 101, f"What you can do: {habit['action']}", CONTENT_WIDTH - 30, 8, True, TEAL_DARK, 11, 2)
        y -= height + 9


def _recommendation_height(item):
    evidence_lines = max(1, len(textwrap.wrap(_plain(item["why_suggested"]), 78)))
    help_lines = max(1, len(textwrap.wrap(_plain(item["why_help"]), 78)))
    action_lines = sum(max(1, len(textwrap.wrap(_plain(action), 72))) for action in item["actions"])
    return 92 + evidence_lines * 11 + help_lines * 11 + action_lines * 11


def _draw_recommendation_pages(pdf, content):
    recommendations = content["recommendations"]
    if not recommendations:
        return
    y = pdf.new_page("Recommended Actions")
    y = _section_title(pdf, y, "Your Recommended Actions", "Prioritised from the strongest well-being factors and recurring habit patterns in your data.")
    prediction = content["prediction"]
    if prediction.get("confidence_band") == "Low confidence":
        message = (
            "The prediction confidence for this analysis is low. The recommendations below are therefore based primarily on the broader patterns found in your wellness and habit data rather than treating the prediction as a definite conclusion."
        )
        pdf.rect(MARGIN, y - 62, CONTENT_WIDTH, 62, AMBER_LIGHT, "E8D2A5")
        pdf.wrapped(MARGIN + 12, y - 18, message, CONTENT_WIDTH - 24, 8, False, AMBER, 11, 4)
        y -= 74
    for index, item in enumerate(recommendations, start=1):
        height = _recommendation_height(item)
        if y - height < 58:
            y = pdf.new_page("Recommended Actions")
        fill = TEAL_LIGHT if item["positive"] else WHITE
        pdf.rect(MARGIN, y - height, CONTENT_WIDTH, height, fill, BORDER)
        pdf.rect(MARGIN + 12, y - 35, 23, 23, TEAL, TEAL, 0)
        pdf.text(MARGIN + 23.5, y - 28, str(index), 9, True, WHITE, "center")
        pdf.text(MARGIN + 45, y - 28, item["title"], 13, True, TEAL_DARK)
        cursor = y - 51
        pdf.text(MARGIN + 14, cursor, "WHY MINDHABITS IS SUGGESTING THIS", 7, True, MUTED)
        cursor = pdf.wrapped(MARGIN + 14, cursor - 14, item["why_suggested"], CONTENT_WIDTH - 28, 8, False, INK, 11)
        pdf.text(MARGIN + 14, cursor - 2, "WHY IT MAY HELP", 7, True, MUTED)
        cursor = pdf.wrapped(MARGIN + 14, cursor - 16, item["why_help"], CONTENT_WIDTH - 28, 8, False, INK, 11)
        pdf.text(MARGIN + 14, cursor - 2, "TRY", 7, True, TEAL)
        cursor -= 16
        for action in item["actions"]:
            pdf.text(MARGIN + 17, cursor, "-", 8, True, TEAL)
            cursor = pdf.wrapped(MARGIN + 28, cursor, action, CONTENT_WIDTH - 42, 8, False, INK, 11)
        y -= height + 10


def _draw_final_page(pdf, content):
    y = pdf.new_page("Next Steps")
    y = _section_title(pdf, y, "What To Focus On Next", "A short summary of the most useful priorities from this report.")
    if content["next_steps"]:
        for index, step in enumerate(content["next_steps"], start=1):
            pdf.rect(MARGIN, y - 52, CONTENT_WIDTH, 52, TEAL_LIGHT, "C7E0DB")
            pdf.text(MARGIN + 14, y - 31, str(index), 14, True, TEAL_DARK)
            pdf.text(MARGIN + 42, y - 30, step, 11, True, INK)
            y -= 62
    else:
        y = pdf.wrapped(MARGIN, y, "No specific action priorities could be identified from the available results.", CONTENT_WIDTH, 10, False, MUTED, 14) - 15

    wellbeing = content["wellbeing"]
    if wellbeing:
        y -= 8
        pdf.text(MARGIN, y, "Technical Information", 12, True, MUTED)
        y -= 18
        best = wellbeing.get("best_model", {})
        metric = ""
        if wellbeing.get("problem_type") == "Classification" and best.get("macro_f1") is not None:
            metric = f"; Macro F1: {best['macro_f1']}%"
        elif best.get("rmse") is not None:
            metric = f"; RMSE: {best['rmse']}"
        pdf.wrapped(MARGIN, y, f"Selected model: {best.get('name', 'Not available')}{metric}. This information is provided as a small appendix for project review.", CONTENT_WIDTH, 8, False, MUTED, 11)

    pdf.rect(MARGIN, 94, CONTENT_WIDTH, 94, SOFT, BORDER)
    pdf.text(MARGIN + 14, 164, "Wellness Disclaimer", 11, True, INK)
    pdf.wrapped(MARGIN + 14, 146, DISCLAIMER, CONTENT_WIDTH - 28, 9, False, MUTED, 13, 5)


def create_wellness_pdf(dataset_name, wellbeing_result, habit_result, generated_at=None, dataframe=None):
    content = build_report_content(dataset_name, wellbeing_result, habit_result, generated_at, dataframe)
    pdf = _PDF()
    _draw_overview_page(pdf, content)
    _draw_factor_page(pdf, content)
    _draw_graph_pages(pdf, content)
    _draw_habits_page(pdf, content)
    _draw_recommendation_pages(pdf, content)
    _draw_final_page(pdf, content)
    return pdf.build()
