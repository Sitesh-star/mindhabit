"""Baseline model comparison for the selected MindHabits outcome."""

import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from utils.recommendation_engine import generate_recommendations
from utils.wellbeing_detector import ID_HINTS, ID_NAMES


MINIMUM_ROWS = 20
MAXIMUM_ROWS = 50_000
RANDOM_STATE = 42
GROUP_COLUMNS = ("user_id", "person_id", "participant_id", "subject_id")

WELLNESS_FACTOR_RULES = (
    (("sleep_duration", "sleep_hours", "hours_slept", "slept_hours"), "Sleep Duration"),
    (("sleep_quality", "quality_of_sleep"), "Sleep Quality"),
    (("sleep_efficiency",), "Sleep Efficiency"),
    (("sleep_latency", "latency"), "Sleep Latency"),
    (("bedtime", "sleep_consistency"), "Sleep Routine"),
    (("wake_episode", "wake_after_sleep", "night_waking", "awaken"), "Wake Episodes"),
    (("sleep", "slept"), "Sleep Pattern"),
    (("screen_time_before_bed", "phone_before_bed", "device_before_bed"), "Screen Time Before Bed"),
    (("screen_time", "phone_usage", "device_usage", "device_use"), "Screen Time"),
    (("caffeine", "coffee"), "Caffeine Intake"),
    (("alcohol",), "Alcohol Intake"),
    (("mindfulness", "meditation", "relaxation"), "Mindfulness"),
    (("exercise", "workout"), "Exercise"),
    (("steps", "step_count"), "Daily Steps"),
    (("activity", "movement", "distance"), "Physical Activity"),
    (("stress", "anxiety"), "Stress Level"),
    (("mental", "emotion", "mood", "wellbeing", "well_being"), "Mental Well-Being"),
    (("hrv", "heart_rate_variability"), "HRV"),
    (("resting_hr", "heart_rate_resting", "resting_heart_rate"), "Resting Heart Rate"),
    (("heart_rate", "heartrate"), "Heart Rate"),
    (("recovery", "rested", "rest", "fatigue", "tired"), "Recovery"),
    (("rem_", "rem_sleep"), "REM Sleep"),
    (("deep_sleep",), "Deep Sleep"),
    (("spo2", "oxygen_saturation"), "Blood Oxygen"),
    (("blood_pressure", "sbp", "dbp"), "Blood Pressure"),
)

CONTEXTUAL_FACTOR_RULES = (
    (("shift_work",), "Shift Work"),
    (("work_hours",), "Work Hours"),
    (("nap_duration", "napping"), "Nap Duration"),
    (("chronotype",), "Chronotype"),
    (("sleep_aid",), "Sleep Aid Use"),
    (("bmi", "body_mass_index"), "BMI"),
    (("weight",), "Weight"),
    (("calories",), "Calories"),
    (("hydration", "water_intake"), "Hydration"),
)

DEMOGRAPHIC_FACTOR_RULES = (
    (("gender", "sex"), "Gender"),
    (("age",), "Age"),
    (("occupation", "employment"), "Occupation"),
    (("region", "country", "ethnicity", "race"), "Background"),
)


class AnalysisError(ValueError):
    """A dataset issue that can be shown safely to the user."""


def _display_name(column_name):
    return str(column_name).replace("_", " ").title()


def _is_target_derived(column_name, target_column):
    column_name = str(column_name).casefold()
    target_column = str(target_column).casefold()
    return (
        column_name.startswith(f"{target_column}_")
        or column_name.endswith(f"_{target_column}")
        or target_column.startswith(f"{column_name}_")
    )


def _is_identifier_like(column_name, series, known_id_columns):
    name = str(column_name).casefold()
    tokens = set(name.split("_"))
    if column_name in known_id_columns:
        return True
    if name in ID_NAMES or name.endswith("_id") or name.startswith("id_"):
        return True

    usable = series.dropna()
    if usable.empty:
        return False
    unique_ratio = usable.nunique(dropna=True) / len(usable)
    return unique_ratio >= 0.95 and bool(tokens & ID_HINTS)


def _prepare_predictors(
    frame, target_column, known_id_columns, leakage_columns=None
):
    predictors = []
    excluded = []
    leakage_columns = set(leakage_columns or [])
    leakage_columns.discard(target_column)

    for column in frame.columns:
        if column == target_column:
            continue

        if column in leakage_columns or _is_target_derived(column, target_column):
            excluded.append(column)
            continue

        series = frame[column]
        usable = series.dropna()
        if usable.empty or usable.nunique(dropna=True) <= 1:
            excluded.append(column)
            continue
        if _is_identifier_like(column, series, known_id_columns):
            excluded.append(column)
            continue

        unique_values = int(usable.nunique(dropna=True))
        unique_ratio = unique_values / len(usable)
        is_text = not pd.api.types.is_numeric_dtype(series)
        if is_text and unique_values > 50 and unique_ratio > 0.5:
            excluded.append(column)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            excluded.append(column)
            continue

        predictors.append(column)

    if not predictors:
        raise AnalysisError(
            "No usable predictor columns were found after removing identifiers, empty fields, and constant values."
        )

    features = frame[predictors].copy()
    for column in predictors:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            features[column] = pd.to_numeric(series, errors="coerce")
            continue

        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() == series.notna().sum() and series.notna().any():
            features[column] = converted
        else:
            features[column] = series.map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )

    features = features.replace([np.inf, -np.inf], np.nan)
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in predictors if column not in numeric_columns
    ]
    return features, numeric_columns, categorical_columns, excluded


def _convert_prediction_features(
    frame, predictor_columns, numeric_columns, categorical_columns
):
    features = frame[predictor_columns].copy()
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in categorical_columns:
        features[column] = features[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )
    return features.replace([np.inf, -np.inf], np.nan)


def _select_prediction_record(frame, predictor_columns):
    usable_features = frame[predictor_columns].notna().any(axis=1)
    if not usable_features.any():
        raise AnalysisError("No record contains enough information for a prediction.")

    for column in frame.columns:
        name = str(column).casefold()
        if "date" not in name and "timestamp" not in name:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce", format="mixed")
        candidates = dates.notna() & usable_features
        if candidates.any():
            index = dates[candidates].idxmax()
            date_label = dates.loc[index].strftime("%d %b %Y")
            return index, f"Latest dated record: {date_label} ({_display_name(column)})"

    index = frame.index[usable_features][-1]
    return index, f"Last usable record in the dataset (row {int(index) + 1})"


def _friendly_class_label(value, available_classes=None):
    text = str(value).strip()
    class_values = {
        str(class_value).strip().casefold()
        for class_value in (
            [] if available_classes is None else available_classes
        )
    }
    if class_values == {"0", "1"}:
        return "Yes" if text == "1" else "No"
    if class_values == {"false", "true"}:
        return "Yes" if text.casefold() == "true" else "No"
    return text.replace("_", " ").replace("-", " ").title()


def _confidence_band(confidence):
    if confidence is None:
        return None
    if confidence < 50:
        return "Low confidence"
    if confidence < 70:
        return "Moderate confidence"
    return "Higher confidence"


def _feature_source(transformed_name, categorical_columns):
    name = str(transformed_name)
    if "__" in name:
        transformer, name = name.split("__", 1)
    else:
        transformer = ""

    if transformer == "categorical":
        for column in sorted(categorical_columns, key=len, reverse=True):
            prefix = f"{column}_"
            if name.startswith(prefix):
                category = name[len(prefix):]
                return column, category
    return name, None


def _matches_factor_keyword(column_name, keyword):
    name = str(column_name).casefold()
    keyword = str(keyword).casefold()
    if len(keyword) <= 4:
        return keyword in name.split("_")
    return keyword in name


def _factor_profile(column_name):
    name = str(column_name).casefold()
    tokens = set(name.split("_"))
    if (
        name in ID_NAMES
        or name.endswith("_id")
        or name.startswith("id_")
        or tokens & {"identifier", "index"}
        or "date" in tokens
        or "timestamp" in name
        or name in {"row", "row_number", "record", "record_number", "time"}
        or "device_model" in name
    ):
        return None

    for keywords, label in WELLNESS_FACTOR_RULES:
        if any(_matches_factor_keyword(name, keyword) for keyword in keywords):
            return {"tier": 1, "label": label}
    for keywords, label in CONTEXTUAL_FACTOR_RULES:
        if any(_matches_factor_keyword(name, keyword) for keyword in keywords):
            return {"tier": 2, "label": label}
    for keywords, label in DEMOGRAPHIC_FACTOR_RULES:
        if any(_matches_factor_keyword(name, keyword) for keyword in keywords):
            return {"tier": 3, "label": label}
    return None


def _extract_contributing_factors(pipeline, categorical_columns, limit=5):
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_, dtype=float)
            importance = (
                np.mean(np.abs(coefficients), axis=0)
                if coefficients.ndim > 1
                else np.abs(coefficients)
            )
        else:
            return []

        if len(feature_names) != len(importance) or not np.isfinite(importance).any():
            return []

        candidates = {1: [], 2: [], 3: []}
        used_columns = set()
        for index in np.argsort(np.nan_to_num(importance, nan=0.0))[::-1]:
            value = float(importance[index])
            if value <= 0:
                continue
            source_column, category = _feature_source(
                feature_names[index], categorical_columns
            )
            if source_column in used_columns:
                continue
            profile = _factor_profile(source_column)
            if profile is None:
                continue
            used_columns.add(source_column)
            display_name = profile["label"]
            if category and category.casefold() not in {"0", "1", "false", "true"}:
                display_name = (
                    f"{display_name}: {_friendly_class_label(category)}"
                )
            candidates[profile["tier"]].append({
                "name": display_name,
                "column": source_column,
                "importance": value,
            })

        meaningful = candidates[1] + candidates[2]
        selected = meaningful[:limit] if meaningful else candidates[3][:min(3, limit)]
        return [
            {"name": factor["name"], "column": factor["column"]}
            for factor in selected
        ]
    except Exception:
        return []


def _make_preprocessor(numeric_columns, categorical_columns, scale_numeric):
    transformers = []
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _find_group_column(frame):
    for column in GROUP_COLUMNS:
        if column not in frame.columns:
            continue
        groups = frame[column].dropna()
        if groups.nunique() >= 2 and groups.duplicated().any():
            return column
    return None


def _split_data(features, target, frame, problem_type, random_state):
    group_column = _find_group_column(frame)
    if group_column:
        groups = frame[group_column].fillna("__missing_group__")
        selected_split = None
        for seed in range(random_state, random_state + 10):
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed
            )
            train_index, test_index = next(splitter.split(features, target, groups))
            if problem_type != "Classification" or (
                target.iloc[train_index].nunique() == target.nunique()
            ):
                selected_split = (train_index, test_index)
                break
        if selected_split is None:
            raise AnalysisError(
                "A reliable group-based split could not preserve all outcome categories."
            )
        train_index, test_index = selected_split
        return (
            features.iloc[train_index], features.iloc[test_index],
            target.iloc[train_index], target.iloc[test_index],
            f"Grouped by {_display_name(group_column)}",
        )

    stratify = None
    if problem_type == "Classification":
        class_counts = target.value_counts()
        test_rows = math.ceil(len(target) * 0.2)
        if class_counts.min() >= 2 and test_rows >= len(class_counts):
            stratify = target

    for seed in range(random_state, random_state + 10):
        split = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=seed,
            stratify=stratify,
        )
        if problem_type != "Classification" or split[2].nunique() == target.nunique():
            return (*split, "Standard 80/20 split")
    raise AnalysisError(
        "A reliable split could not preserve all outcome categories."
    )


def _classification_models(class_count):
    xgb_parameters = {
        "n_estimators": 80,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
        "verbosity": 0,
    }
    if class_count == 2:
        xgb_parameters.update(objective="binary:logistic", eval_metric="logloss")
    else:
        xgb_parameters.update(
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
        )

    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000), True),
        ("Decision Tree Classifier", DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=2, random_state=RANDOM_STATE
        ), False),
        ("Random Forest Classifier", RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ), False),
        ("XGBoost Classifier", XGBClassifier(**xgb_parameters), False),
    ]


def _regression_models():
    return [
        ("Linear Regression", LinearRegression(), True),
        ("Decision Tree Regressor", DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=2, random_state=RANDOM_STATE
        ), False),
        ("Random Forest Regressor", RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ), False),
        ("XGBoost Regressor", XGBRegressor(
            objective="reg:squarederror", n_estimators=80, max_depth=4,
            learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=2, verbosity=0
        ), False),
    ]


def _classification_metrics(actual, predicted, labels):
    accuracy = accuracy_score(actual, predicted)
    macro_precision = precision_score(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        actual, predicted, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(accuracy * 100, 1),
        "macro_precision": round(macro_precision * 100, 1),
        "macro_recall": round(macro_recall * 100, 1),
        "macro_f1": round(macro_f1 * 100, 1),
        "weighted_f1": round(weighted_f1 * 100, 1),
        "_selection_primary": macro_f1,
        "_selection_secondary": accuracy,
    }


def _regression_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = r2_score(actual, predicted)
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "_selection_primary": rmse,
        "_selection_secondary": r2,
    }


def analyze_models(
    df,
    target_column,
    problem_type,
    id_columns=None,
    leakage_columns=None,
    random_state=RANDOM_STATE,
):
    """Train four baseline pipelines and return their held-out performance."""
    if target_column not in df.columns:
        raise AnalysisError("The recommended outcome is no longer in this dataset.")
    if problem_type not in {"Classification", "Regression"}:
        raise AnalysisError("The analysis type could not be determined.")

    df = df.reset_index(drop=True).copy()
    total_rows = len(df)
    target_series = df[target_column]
    if problem_type == "Regression":
        target_series = pd.to_numeric(target_series, errors="coerce")
    valid_target = target_series.notna()
    usable_rows = int(valid_target.sum())

    if not total_rows or usable_rows / total_rows < 0.3:
        raise AnalysisError(
            "The recommended outcome has too many missing values for a reliable comparison."
        )
    if usable_rows < MINIMUM_ROWS:
        raise AnalysisError(
            f"At least {MINIMUM_ROWS} usable rows are needed to compare models."
        )

    working_frame = df.loc[valid_target].copy()
    target = target_series.loc[valid_target].copy()
    if problem_type == "Classification":
        target = target.astype(str)
        class_counts = target.value_counts()
        if len(class_counts) < 2:
            raise AnalysisError("The recommended outcome contains only one category.")
        if len(class_counts) > 50:
            raise AnalysisError(
                "The recommended outcome has too many categories for this baseline comparison."
            )
        if class_counts.min() < 2:
            raise AnalysisError(
                "Each outcome category needs at least two usable records."
            )
    elif target.nunique(dropna=True) < 2:
        raise AnalysisError("The recommended outcome does not contain enough variation.")

    features, numeric_columns, categorical_columns, excluded = _prepare_predictors(
        working_frame,
        target_column,
        set(id_columns or []),
        leakage_columns=leakage_columns,
    )
    predictor_columns = numeric_columns + categorical_columns
    prediction_index, prediction_record = _select_prediction_record(
        df, predictor_columns
    )
    prediction_features = _convert_prediction_features(
        df.loc[[prediction_index]],
        predictor_columns,
        numeric_columns,
        categorical_columns,
    )

    # Keep the displayed record out of all model training and evaluation.
    if prediction_index in working_frame.index:
        working_frame = working_frame.drop(index=prediction_index)
        target = target.drop(index=prediction_index)
        features = features.drop(index=prediction_index)

    if len(working_frame) < MINIMUM_ROWS:
        raise AnalysisError(
            f"At least {MINIMUM_ROWS} training rows are needed after holding out the prediction record."
        )
    if problem_type == "Classification" and target.value_counts().min() < 2:
        raise AnalysisError(
            "Each outcome category needs at least two training records after holding out the prediction record."
        )

    sampled = False
    if len(working_frame) > MAXIMUM_ROWS:
        sampled = True
        if problem_type == "Classification":
            sample_index, _ = train_test_split(
                working_frame.index,
                train_size=MAXIMUM_ROWS,
                random_state=random_state,
                stratify=target,
            )
        else:
            sample_index = working_frame.sample(
                n=MAXIMUM_ROWS, random_state=random_state
            ).index
        working_frame = working_frame.loc[sample_index]
        target = target.loc[sample_index]
        features = features.loc[sample_index]

    if problem_type == "Classification":
        encoder = LabelEncoder()
        encoded_target = pd.Series(
            encoder.fit_transform(target), index=target.index, name=target.name
        )
        target = encoded_target
        class_count = len(encoder.classes_)
        model_specs = _classification_models(class_count)
    else:
        class_count = None
        model_specs = _regression_models()

    x_train, x_test, y_train, y_test, split_method = _split_data(
        features, target, working_frame, problem_type, random_state
    )
    if len(x_train) < 2 or len(x_test) < 1:
        raise AnalysisError("The dataset is too small for an 80/20 comparison.")

    model_results = []
    model_errors = []
    fitted_pipelines = {}
    for model_name, estimator, scale_numeric in model_specs:
        preprocessor = _make_preprocessor(
            numeric_columns, categorical_columns, scale_numeric
        )
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ])
        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            if problem_type == "Classification":
                metrics = _classification_metrics(
                    y_test, predictions, labels=np.arange(class_count)
                )
            else:
                metrics = _regression_metrics(y_test, predictions)
            model_results.append({"name": model_name, **metrics})
            fitted_pipelines[model_name] = pipeline
        except Exception:
            model_errors.append(model_name)

    if not model_results:
        raise AnalysisError(
            "The models could not be compared with this dataset. Check the available values and try again."
        )

    if problem_type == "Classification":
        model_results.sort(key=lambda result: (
            -result["_selection_primary"], -result["_selection_secondary"]
        ))
    else:
        model_results.sort(key=lambda result: (
            result["_selection_primary"], -result["_selection_secondary"]
        ))

    best_model_name = model_results[0]["name"]
    best_pipeline = fitted_pipelines[best_model_name]
    raw_prediction = best_pipeline.predict(prediction_features)[0]

    if problem_type == "Classification":
        encoded_prediction = int(raw_prediction)
        original_label = encoder.inverse_transform([encoded_prediction])[0]
        prediction_value = _friendly_class_label(original_label, encoder.classes_)
        confidence = None
        if hasattr(best_pipeline, "predict_proba"):
            probabilities = best_pipeline.predict_proba(prediction_features)[0]
            confidence = round(float(np.max(probabilities)) * 100, 1)
    else:
        numeric_prediction = round(float(raw_prediction), 2)
        prediction_value = f"{numeric_prediction:.2f}".rstrip("0").rstrip(".")
        confidence = None

    contributing_factors = _extract_contributing_factors(
        best_pipeline, categorical_columns, limit=5
    )
    recommendations = generate_recommendations(
        df,
        prediction_index,
        contributing_factors,
        limit=5,
    )

    for model_result in model_results:
        model_result.pop("_selection_primary", None)
        model_result.pop("_selection_secondary", None)

    return {
        "outcome_name": target_column,
        "outcome_display_name": _display_name(target_column),
        "problem_type": problem_type,
        "best_model": model_results[0],
        "models": model_results,
        "prediction": {
            "outcome_label": f"Predicted {_display_name(target_column)}",
            "value": prediction_value,
            "confidence": confidence,
            "confidence_band": _confidence_band(confidence),
            "record_description": prediction_record,
        },
        "contributing_factors": contributing_factors,
        "recommendations": recommendations,
        "rows_used": len(working_frame),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "total_usable_rows": usable_rows,
        "predictor_count": len(numeric_columns) + len(categorical_columns),
        "excluded_predictors": excluded,
        "split_method": split_method,
        "sampled": sampled,
        "model_errors": model_errors,
    }
