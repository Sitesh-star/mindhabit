from flask import Flask, render_template, request, redirect, url_for, jsonify, abort, send_file, session
import csv
import hashlib
import hmac
import io
import json
import os
from uuid import uuid4
import pandas as pd
from werkzeug.utils import secure_filename
from utils.wellbeing_detector import detect_wellbeing_columns
from utils.ml_analyzer import AnalysisError, MINIMUM_ROWS, analyze_models
from utils.habit_insights import analyze_habit_insights
from utils.export_manager import create_csv_report, create_pdf_report
from utils.dashboard_analyzer import (
    build_dashboard_content,
    prepare_dashboard_dataframe,
)
from utils.dataset_storage import (
    DatasetStorageError, InvalidDatasetError, blob_storage_enabled,
    create_dataset_path, dataset_filename, dataset_identity, load_dataset,
    make_blob_reference, save_local_upload, validate_blob_reference,
)
from utils.database import Database, DatabaseError, resolve_database_url


app = Flask(__name__)
app.config["DATABASE"] = os.path.join(app.instance_path, "mindhabits.sqlite3")
app.config["DATABASE_URL"], database_source = resolve_database_url(
    os.environ,
    app.config["DATABASE"],
    production=bool(os.environ.get("VERCEL")),
)
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
if os.environ.get("VERCEL") and not os.environ.get("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be configured for production.")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("VERCEL"))
if database_source == "neon":
    app.logger.info("Using PostgreSQL database from Neon environment configuration.")
elif database_source == "postgresql":
    app.logger.info("Using PostgreSQL database from DATABASE_URL configuration.")



def _database():
    database_url = app.config.get("DATABASE_URL")
    if not database_url:
        database_url = f"sqlite:///{app.config['DATABASE']}"
    return Database(database_url)


def _visitor_id():
    return session["visitor_id"]


@app.before_request
def ensure_anonymous_visitor():
    if "visitor_id" not in session:
        session["visitor_id"] = str(uuid4())


def _active_dataset_reference():
    active = session.get("active_dataset")
    if isinstance(active, dict) and active.get("source") in {"blob", "manual"}:
        return active
    # Keep existing browser sessions usable after deployment of the unified state.
    legacy_reference = session.get("dataset_reference")
    if isinstance(legacy_reference, dict):
        return {"source": "blob", "reference": legacy_reference}
    return None


def _set_active_blob(reference):
    session["active_dataset"] = {"source": "blob", "reference": reference}
    session["dataset_reference"] = reference
    session.pop("current_spreadsheet_id", None)


def _set_active_manual(spreadsheet_id):
    session["active_dataset"] = {
        "source": "manual", "spreadsheet_id": int(spreadsheet_id)
    }
    session.pop("dataset_reference", None)
    session.pop("current_spreadsheet_id", None)


def load_active_dataset():
    """Load the one dataset selected for every preview and analysis page."""
    active = _active_dataset_reference()
    if active is None:
        return None
    if active["source"] == "blob":
        reference = active.get("reference")
        frame = load_dataset(reference, app.config["UPLOAD_FOLDER"])
        return {
            "dataframe": frame,
            "identity": dataset_identity(reference),
            "name": dataset_filename(reference),
            "source": "blob",
            "source_label": "CSV upload",
        }

    record = _database().get_spreadsheet(
        _visitor_id(), active.get("spreadsheet_id")
    )
    if record is None:
        raise DatasetStorageError("The selected manual spreadsheet no longer exists.")
    frame = pd.DataFrame(record["rows"], columns=record["columns"])
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    version_payload = json.dumps(
        {"columns": record["columns"], "rows": record["rows"]},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    version = hashlib.sha256(version_payload).hexdigest()[:16]
    return {
        "dataframe": frame,
        "identity": f"manual:{record['id']}:{version}",
        "name": record["name"],
        "source": "manual",
        "source_label": "Manual entry",
        "spreadsheet_id": record["id"],
    }


def _dataset_identity():
    active_reference = _active_dataset_reference()
    if active_reference is None:
        return None
    if active_reference["source"] == "blob":
        return dataset_identity(active_reference.get("reference"))
    active = load_active_dataset()
    return active["identity"]


def _load_current_dataset():
    active = load_active_dataset()
    if active is None:
        raise DatasetStorageError("No dataset is currently selected.")
    return active["dataframe"]


def _dataset_load_failed():
    session.pop("active_dataset", None)
    session.pop("dataset_reference", None)
    session.pop("current_spreadsheet_id", None)
    return render_template(
        "empty_dataset.html", page_title="Data unavailable",
        active_page="upload",
        message="The active dataset could not be loaded. Upload a CSV file or enter your data manually to continue.",
    ), 404


def _no_data_page(page_title, active_page):
    return render_template(
        "empty_dataset.html", page_title=page_title, active_page=active_page,
        message="Upload a CSV file or enter your data manually to continue.",
    )


def _direct_blob_upload_enabled():
    """Production must never fall back to sending dataset bytes through Flask."""
    return bool(os.environ.get("VERCEL")) or blob_storage_enabled()


def _reset_analysis_results():
    # Results are keyed by dataset, so a new dataset automatically starts clean.
    return None


def _store_analysis_result(result_type, result):
    dataset_id = _dataset_identity()
    if not dataset_id:
        raise DatabaseError("No active dataset is available for this analysis.")
    _database().save_analysis_result(_visitor_id(), dataset_id, result_type, result)


def _current_analysis_results():
    return _database().get_analysis_results(_visitor_id(), _dataset_identity())


@app.errorhandler(DatabaseError)
def database_unavailable(error):
    app.logger.warning("Database operation failed: %s", error)
    return "MindHabits storage is temporarily unavailable. Please try again.", 503


# --------------------------------------------------
# Home
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# --------------------------------------------------
# Manual Entry
# --------------------------------------------------
@app.route("/manual")
@app.route("/manual/<int:spreadsheet_id>")
def manual(spreadsheet_id=None):
    database = _database()
    spreadsheet = None
    if spreadsheet_id is not None:
        spreadsheet = database.get_spreadsheet(_visitor_id(), spreadsheet_id)
        if spreadsheet is None:
            abort(404)
        _set_active_manual(spreadsheet_id)
    saved_spreadsheets = database.list_spreadsheets(_visitor_id())
    return render_template(
        "manual.html", spreadsheet=spreadsheet, saved_spreadsheets=saved_spreadsheets
    )


@app.route("/api/spreadsheets", methods=["POST"])
@app.route("/api/spreadsheets/<int:spreadsheet_id>", methods=["PUT"])
def save_spreadsheet(spreadsheet_id=None):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a spreadsheet as JSON."), 400
    name = payload.get("name")
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(name, str) or not name.strip():
        return jsonify(error="Please enter a spreadsheet name."), 400
    if not isinstance(columns, list) or not columns or not all(
        isinstance(column, str) and column.strip() for column in columns
    ):
        return jsonify(error="Each column needs a name."), 400
    columns = [column.strip() for column in columns]
    if len({column.casefold() for column in columns}) != len(columns):
        return jsonify(error="Column names must be unique."), 400
    if not isinstance(rows, list) or not rows or not all(
        isinstance(row, list) and len(row) == len(columns)
        and all(isinstance(value, str) for value in row) for row in rows
    ):
        return jsonify(error="Each row must contain one text value per column."), 400

    database = _database()
    if database.spreadsheet_name_exists(
        _visitor_id(), name.strip(), exclude_id=spreadsheet_id
    ):
        return jsonify(
            error="A spreadsheet with this name already exists. Please choose another name."
        ), 409
    if spreadsheet_id is None:
        spreadsheet_id = database.create_spreadsheet(
            _visitor_id(), name.strip(), columns, rows
        )
        status = 201
    else:
        if not database.update_spreadsheet(
            _visitor_id(), spreadsheet_id, name.strip(), columns, rows
        ):
            abort(404)
        status = 200
    _set_active_manual(spreadsheet_id)
    return jsonify(
        id=spreadsheet_id,
        name=name.strip(),
        url=url_for("manual", spreadsheet_id=spreadsheet_id),
        save_url=url_for("save_spreadsheet", spreadsheet_id=spreadsheet_id, _method="PUT"),
        download_url=url_for("download_spreadsheet", spreadsheet_id=spreadsheet_id),
        preview_url=url_for("preview"),
        dashboard_url=url_for("dashboard")
    ), status


@app.route("/manual/<int:spreadsheet_id>/delete", methods=["POST"])
def delete_spreadsheet(spreadsheet_id):
    database = _database()
    if not database.delete_spreadsheet(_visitor_id(), spreadsheet_id):
        abort(404)
    database.delete_analysis_results(
        _visitor_id(), f"manual:{spreadsheet_id}:"
    )
    active = _active_dataset_reference()
    if (
        active and active.get("source") == "manual"
        and active.get("spreadsheet_id") == spreadsheet_id
    ):
        session.pop("active_dataset", None)
    return redirect(url_for("manual"))


@app.route("/manual/<int:spreadsheet_id>/download")
def download_spreadsheet(spreadsheet_id):
    record = _database().get_spreadsheet(_visitor_id(), spreadsheet_id)
    if record is None:
        abort(404)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(record["columns"])
    writer.writerows(record["rows"])
    filename = secure_filename(record["name"]) or "spreadsheet"
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        mimetype="text/csv", as_attachment=True, download_name=f"{filename}.csv"
    )


# --------------------------------------------------
# Upload
# --------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if _direct_blob_upload_enabled():
            return render_template("upload.html", blob_upload_enabled=True,
                upload_error="Use the secure direct upload to upload this dataset."), 400
        try:
            reference = save_local_upload(
                request.files.get("file"), app.config["UPLOAD_FOLDER"]
            )
        except AttributeError:
            return render_template(
                "upload.html", blob_upload_enabled=False,
                upload_error="Invalid file format. Please choose a CSV file.",
            ), 400
        except InvalidDatasetError as error:
            return render_template("upload.html", blob_upload_enabled=False,
                                   upload_error=str(error)), 400
        _set_active_blob(reference)
        _reset_analysis_results()
        return redirect(url_for("preview"))

    return render_template("upload.html", blob_upload_enabled=_direct_blob_upload_enabled(),
                           upload_error=request.args.get("error"))


@app.route("/upload/prepare", methods=["POST"])
def prepare_blob_upload():
    if not blob_storage_enabled():
        return jsonify(error="Direct Blob upload is not configured."), 400
    payload = request.get_json(silent=True) or {}
    filename = secure_filename(str(payload.get("filename", "")))
    size = payload.get("size")
    content_type = payload.get("content_type")
    if not filename.lower().endswith(".csv") or not isinstance(size, int):
        return jsonify(error="Invalid file format. Please choose a CSV file."), 400
    if content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
        return jsonify(error="Invalid file format. Please choose a CSV file."), 400
    if size <= 0:
        return jsonify(error="The selected CSV file is empty."), 400
    if size > 25 * 1024 * 1024:
        return jsonify(error="File is too large. Please choose a CSV file smaller than 25 MB."), 400
    pathname = create_dataset_path(filename)
    session["pending_dataset_upload"] = {"pathname": pathname, "filename": filename}
    secret = app.config["SECRET_KEY"]
    if isinstance(secret, str):
        secret = secret.encode()
    claim = hmac.new(secret, f"{pathname}:{size}".encode(), hashlib.sha256).hexdigest()
    return jsonify(
        pathname=pathname, filename=filename, size=size,
        contentType="text/csv", claim=claim
    )


@app.route("/upload/complete", methods=["POST"])
def complete_blob_upload():
    payload = request.get_json(silent=True) or {}
    pending = session.pop("pending_dataset_upload", None)
    requested_pathname = payload.get("requested_pathname")
    canonical_pathname = payload.get("pathname")
    canonical_url = payload.get("url")
    expected_folder = f"{pending['pathname'].rsplit('/', 1)[0]}/" if pending else None
    if (
        not pending
        or payload.get("storage") != "blob"
        or requested_pathname != pending.get("pathname")
        or not isinstance(canonical_pathname, str)
        or not canonical_pathname.startswith(expected_folder)
        or payload.get("filename") != pending.get("filename")
    ):
        return jsonify(error="The upload authorization is missing or expired."), 400
    reference = make_blob_reference(
        canonical_pathname, pending["filename"], canonical_url
    )
    app.logger.info("Received canonical uploaded Blob pathname: %s", canonical_pathname)
    try:
        validate_blob_reference(reference)
        load_dataset(reference, app.config["UPLOAD_FOLDER"])
    except InvalidDatasetError:
        return jsonify(error="MindHabits could not read this CSV file. Please check the file and try again."), 400
    except DatasetStorageError:
        app.logger.warning("Uploaded Blob reference could not be verified.", exc_info=True)
        return jsonify(error="The dataset could not be uploaded. Please try again."), 400
    app.logger.info("Storing canonical Blob pathname in session: %s", canonical_pathname)
    _set_active_blob(reference)
    _reset_analysis_results()
    return jsonify(preview_url=url_for("preview"))


# --------------------------------------------------
# Data Preview
# --------------------------------------------------
@app.route("/preview")
def preview():
    active = _active_dataset_reference()
    if active is None:
        return _no_data_page("Data Preview", "preview")
    try:
        dataset = load_active_dataset()
        df = dataset["dataframe"]
    except DatasetStorageError:
        app.logger.warning("The current dataset could not be loaded.", exc_info=True)
        return _dataset_load_failed()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    columns = df.columns.tolist()

    rows = (
        df.head(5)
        .fillna("")
        .to_dict(
            orient="records"
        )
    )

    total_rows = len(df)

    total_columns = len(
        df.columns
    )

    missing_values = int(
        df.isnull()
        .sum()
        .sum()
    )

    duplicate_rows = int(
        df.duplicated()
        .sum()
    )

    filename = dataset["name"]

    return render_template(
        "preview.html",
        columns=columns,
        rows=rows,
        total_rows=total_rows,
        total_columns=total_columns,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        filename=filename,
        source_label=dataset["source_label"],
    )


# --------------------------------------------------
# Dashboard
# --------------------------------------------------
@app.route("/dashboard")
def dashboard():
    spreadsheet_id = request.args.get("spreadsheet_id", type=int)
    if spreadsheet_id is not None:
        if _database().get_spreadsheet(_visitor_id(), spreadsheet_id) is None:
            abort(404)
        _set_active_manual(spreadsheet_id)
    if _active_dataset_reference() is None:
        return _no_data_page("Dashboard", "dashboard")
    try:
        dataset = load_active_dataset()
        df = dataset["dataframe"]
    except DatasetStorageError:
        return _dataset_load_failed()
    dataset_name = dataset["name"]
    source_label = dataset["source_label"]

    df = prepare_dashboard_dataframe(df)

    total_records = len(df)
    total_columns = len(df.columns)
    total_cells = total_records * total_columns
    missing_cells = int(df.isna().sum().sum())
    completion_rate = round(
        ((total_cells - missing_cells) / total_cells) * 100
    ) if total_cells else 0
    duplicate_rows = int(df.duplicated().sum())

    dashboard_content = build_dashboard_content(df)

    return render_template(
        "dashboard.html",
        dataset_name=dataset_name,
        source_label=source_label,
        total_records=total_records,
        total_columns=total_columns,
        missing_cells=missing_cells,
        completion_rate=completion_rate,
        duplicate_rows=duplicate_rows,
        column_summaries=dashboard_content["column_summaries"],
        distribution_charts=dashboard_content["distribution_charts"],
        wellness_detected=dashboard_content["wellness_detected"],
        wellness_column_count=dashboard_content["wellness_column_count"],
        date_column=dashboard_content["date_column"],
        selection_note=dashboard_content["selection_note"],
        dashboard_url=url_for("dashboard")
    )


# --------------------------------------------------
# Mental Well-Being
# --------------------------------------------------
@app.route("/wellbeing", methods=["GET", "POST"])
def wellbeing():
    if _active_dataset_reference() is None:
        return _no_data_page("Mental Well-Being", "wellbeing")
    try:
        df = _load_current_dataset()
    except DatasetStorageError:
        return _dataset_load_failed()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    detection = detect_wellbeing_columns(df)
    recommended_outcome = detection["recommended_outcome"]
    analysis_ready = bool(
        recommended_outcome
        and len(df) >= MINIMUM_ROWS
        and df.dropna(how="all").shape[0] >= MINIMUM_ROWS
        and len(df.columns) >= 2
    )
    analysis_result = None
    analysis_error = None

    if request.method == "POST":
        if not analysis_ready:
            analysis_error = (
                "MindHabits found some relevant wellness information, but there are not enough usable records or predictors to run a reliable analysis."
            )
        else:
            try:
                analysis_result = analyze_models(
                    df,
                    target_column=recommended_outcome["name"],
                    problem_type=recommended_outcome["problem_type"],
                    id_columns=detection["id_columns"],
                    leakage_columns=[
                        outcome["name"]
                        for outcome in detection["possible_outcomes"]
                    ],
                )
                _store_analysis_result("wellbeing", analysis_result)
            except AnalysisError as error:
                analysis_error = str(error)
            except Exception:
                app.logger.exception("Well-being model analysis failed")
                analysis_error = (
                    "MindHabits could not complete the model comparison for this dataset."
                )

    return render_template(
        "wellbeing.html",
        relevant_columns=detection["relevant_columns"],
        grouped_columns=detection["grouped_columns"],
        possible_outcomes=detection["possible_outcomes"],
        recommended_outcome=recommended_outcome,
        alternative_outcomes=detection["alternative_outcomes"],
        total_relevant_columns=detection["total_relevant_columns"],
        analysis_result=analysis_result,
        analysis_error=analysis_error,
        analysis_ready=analysis_ready,
    )


# --------------------------------------------------
# Habit Insights
# --------------------------------------------------
@app.route("/habits")
def habits():
    if _active_dataset_reference() is None:
        return _no_data_page("Habit Insights", "habits")
    try:
        df = _load_current_dataset()
    except DatasetStorageError:
        return _dataset_load_failed()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )
    habit_result = analyze_habit_insights(df)
    _store_analysis_result("habits", habit_result)

    return render_template("habits.html", habit_result=habit_result)


# --------------------------------------------------
# Export Results
# --------------------------------------------------
@app.route("/export")
def export():
    if _active_dataset_reference() is None:
        return _no_data_page("Export Results", "export")
    try:
        dataset = load_active_dataset()
    except DatasetStorageError:
        return _dataset_load_failed()
    results = _current_analysis_results()
    wellbeing_result = results["wellbeing"]
    habit_result = results["habits"]
    habit_available = bool(habit_result and habit_result.get("insights"))
    has_results = bool(wellbeing_result or habit_available)
    dataset_name = dataset["name"]
    return render_template(
        "export.html",
        wellbeing_result=wellbeing_result,
        habit_result=habit_result if habit_available else None,
        habit_available=habit_available,
        has_results=has_results,
        dataset_name=dataset_name,
    )


@app.route("/export/csv")
def download_analysis_csv():
    if _active_dataset_reference() is None:
        return redirect(url_for("export"))
    results = _current_analysis_results()
    habit_result = results["habits"]
    if not results["wellbeing"] and not (
        habit_result and habit_result.get("insights")
    ):
        return redirect(url_for("export"))

    try:
        dataset_name = load_active_dataset()["name"]
    except DatasetStorageError:
        return _dataset_load_failed()
    report = create_csv_report(
        dataset_name,
        results["wellbeing"],
        habit_result,
    )
    return send_file(
        io.BytesIO(report),
        mimetype="text/csv",
        as_attachment=True,
        download_name="mindhabits_analysis_report.csv",
    )


@app.route("/export/pdf")
def download_analysis_pdf():
    if _active_dataset_reference() is None:
        return redirect(url_for("export"))
    results = _current_analysis_results()
    habit_result = results["habits"]
    if not results["wellbeing"] and not (
        habit_result and habit_result.get("insights")
    ):
        return redirect(url_for("export"))

    try:
        dataset = load_active_dataset()
    except DatasetStorageError:
        return _dataset_load_failed()
    dataset_name = dataset["name"]
    report_dataframe = dataset["dataframe"]
    report = create_pdf_report(
        dataset_name,
        results["wellbeing"],
        habit_result,
        dataframe=report_dataframe,
    )
    return send_file(
        io.BytesIO(report),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="mindhabits_analysis_report.pdf",
    )


# --------------------------------------------------
# Help / About
# --------------------------------------------------
@app.route("/help")
def help_page():
    return render_template(
        "help.html"
    )


# --------------------------------------------------
# Run Application
# --------------------------------------------------
if __name__ == "__main__":

    app.run(
        debug=True
    )
