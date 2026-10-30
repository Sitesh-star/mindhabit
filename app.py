from flask import Flask, render_template, request, redirect, url_for, jsonify, abort, send_file, session
import csv
import hashlib
import hmac
import io
import os
from uuid import uuid4
import pandas as pd
from werkzeug.utils import secure_filename
from utils.wellbeing_detector import detect_wellbeing_columns
from utils.ml_analyzer import AnalysisError, analyze_models
from utils.habit_insights import analyze_habit_insights
from utils.export_manager import create_csv_report, create_pdf_report
from utils.dashboard_analyzer import (
    build_dashboard_content,
    prepare_dashboard_dataframe,
)
from utils.dataset_storage import (
    DatasetStorageError, InvalidDatasetError, blob_storage_enabled,
    create_dataset_path, dataset_filename, dataset_identity, load_dataset,
    make_blob_reference, save_local_upload,
)
from utils.database import Database, DatabaseError


app = Flask(__name__)
app.config["DATABASE"] = os.path.join(app.instance_path, "mindhabits.sqlite3")
app.config["DATABASE_URL"] = os.environ.get("DATABASE_URL")
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
if os.environ.get("VERCEL") and not os.environ.get("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be configured for production.")
if os.environ.get("VERCEL") and not os.environ.get("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL must be configured for production.")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("VERCEL"))



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


def _dataset_identity():
    return dataset_identity(session.get("dataset_reference"))


def _dataset_reference():
    return session.get("dataset_reference")


def _load_current_dataset():
    return load_dataset(_dataset_reference(), app.config["UPLOAD_FOLDER"])


def _dataset_load_failed():
    session.pop("dataset_reference", None)
    return redirect(url_for("upload", error="The dataset could not be loaded. Please upload it again."))


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
        session["current_spreadsheet_id"] = spreadsheet_id
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
    session["current_spreadsheet_id"] = spreadsheet_id
    return jsonify(
        id=spreadsheet_id,
        name=name.strip(),
        url=url_for("manual", spreadsheet_id=spreadsheet_id),
        save_url=url_for("save_spreadsheet", spreadsheet_id=spreadsheet_id, _method="PUT"),
        download_url=url_for("download_spreadsheet", spreadsheet_id=spreadsheet_id),
        dashboard_url=url_for("dashboard", spreadsheet_id=spreadsheet_id)
    ), status


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
        if blob_storage_enabled():
            return render_template("upload.html", blob_upload_enabled=True,
                upload_error="Use the secure direct upload to upload this dataset."), 400
        try:
            reference = save_local_upload(
                request.files.get("file"), app.config["UPLOAD_FOLDER"]
            )
        except (AttributeError, InvalidDatasetError) as error:
            return render_template("upload.html", blob_upload_enabled=False,
                                   upload_error=str(error)), 400
        session["dataset_reference"] = reference
        session.pop("current_spreadsheet_id", None)
        _reset_analysis_results()
        return redirect(url_for("preview"))

    return render_template("upload.html", blob_upload_enabled=blob_storage_enabled(),
                           upload_error=request.args.get("error"))


@app.route("/upload/prepare", methods=["POST"])
def prepare_blob_upload():
    if not blob_storage_enabled():
        return jsonify(error="Direct Blob upload is not configured."), 400
    payload = request.get_json(silent=True) or {}
    filename = secure_filename(str(payload.get("filename", "")))
    size = payload.get("size")
    if not filename.lower().endswith(".csv") or not isinstance(size, int):
        return jsonify(error="Please choose a valid CSV file."), 400
    if size <= 0 or size > 25 * 1024 * 1024:
        return jsonify(error="The CSV must be non-empty and no larger than 25 MB."), 400
    pathname = create_dataset_path(filename)
    session["pending_dataset_upload"] = {"pathname": pathname, "filename": filename}
    secret = app.config["SECRET_KEY"]
    if isinstance(secret, str):
        secret = secret.encode()
    claim = hmac.new(secret, f"{pathname}:{size}".encode(), hashlib.sha256).hexdigest()
    return jsonify(pathname=pathname, size=size, claim=claim)


@app.route("/upload/complete", methods=["POST"])
def complete_blob_upload():
    payload = request.get_json(silent=True) or {}
    pending = session.pop("pending_dataset_upload", None)
    if not pending or payload.get("pathname") != pending.get("pathname"):
        return jsonify(error="The upload authorization is missing or expired."), 400
    session["dataset_reference"] = make_blob_reference(
        pending["pathname"], pending["filename"]
    )
    session.pop("current_spreadsheet_id", None)
    _reset_analysis_results()
    return jsonify(preview_url=url_for("preview"))


# --------------------------------------------------
# Data Preview
# --------------------------------------------------
@app.route("/preview")
def preview():
    if _dataset_reference() is None:
        return redirect(url_for("upload"))
    try:
        df = _load_current_dataset()
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

    filename = dataset_filename(_dataset_reference())

    return render_template(
        "preview.html",
        columns=columns,
        rows=rows,
        total_rows=total_rows,
        total_columns=total_columns,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        filename=filename
    )


# --------------------------------------------------
# Dashboard
# --------------------------------------------------
@app.route("/dashboard")
def dashboard():
    spreadsheet_id = request.args.get("spreadsheet_id", type=int)
    if spreadsheet_id is None:
        spreadsheet_id = session.get("current_spreadsheet_id")

    if spreadsheet_id is not None:
        record = _database().get_spreadsheet(_visitor_id(), spreadsheet_id)
        if record is None:
            abort(404)
        df = pd.DataFrame(record["rows"], columns=record["columns"])
        dataset_name = record["name"]
        source_label = "Manual entry"
    else:
        if _dataset_reference() is None:
            return redirect(url_for("upload"))
        try:
            df = _load_current_dataset()
        except DatasetStorageError:
            return _dataset_load_failed()
        dataset_name = dataset_filename(_dataset_reference())
        source_label = "CSV upload"

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
        dashboard_url=url_for("dashboard", spreadsheet_id=spreadsheet_id)
    )


# --------------------------------------------------
# Mental Well-Being
# --------------------------------------------------
@app.route("/wellbeing", methods=["GET", "POST"])
def wellbeing():
    if _dataset_reference() is None:
        return redirect(url_for("upload"))
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
    analysis_result = None
    analysis_error = None

    if request.method == "POST":
        if recommended_outcome is None:
            analysis_error = (
                "MindHabits could not identify a suitable outcome for model comparison."
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
        analysis_error=analysis_error
    )


# --------------------------------------------------
# Habit Insights
# --------------------------------------------------
@app.route("/habits")
def habits():
    habit_result = None
    if _dataset_reference() is not None:
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
        habit_result = analyze_habit_insights(df)
        _store_analysis_result("habits", habit_result)

    return render_template("habits.html", habit_result=habit_result)


# --------------------------------------------------
# Export Results
# --------------------------------------------------
@app.route("/export")
def export():
    results = _current_analysis_results()
    wellbeing_result = results["wellbeing"]
    habit_result = results["habits"]
    habit_available = bool(habit_result and habit_result.get("insights"))
    has_results = bool(wellbeing_result or habit_available)
    dataset_name = (
        dataset_filename(_dataset_reference()) if _dataset_reference() else None
    )
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
    results = _current_analysis_results()
    habit_result = results["habits"]
    if not results["wellbeing"] and not (
        habit_result and habit_result.get("insights")
    ):
        return redirect(url_for("export"))

    dataset_name = dataset_filename(_dataset_reference()) if _dataset_reference() else None
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
    results = _current_analysis_results()
    habit_result = results["habits"]
    if not results["wellbeing"] and not (
        habit_result and habit_result.get("insights")
    ):
        return redirect(url_for("export"))

    dataset_name = dataset_filename(_dataset_reference()) if _dataset_reference() else None
    report_dataframe = None
    if _dataset_reference():
        try:
            report_dataframe = _load_current_dataset()
        except DatasetStorageError:
            app.logger.warning(
                "The current dataset could not be read for PDF charts.",
                exc_info=True,
            )
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
