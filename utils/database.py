"""Small persistence layer for PostgreSQL production and SQLite development."""

from contextlib import contextmanager
from datetime import date, datetime
import json
import math
import os
import sqlite3


class DatabaseError(Exception):
    """Persistent storage is unavailable or contains invalid data."""


def json_safe(value):
    """Convert analysis output, including NumPy/pandas scalars, to JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


class Database:
    def __init__(self, database_url):
        self.database_url = database_url
        self.postgres = database_url.startswith(("postgres://", "postgresql://"))

    @contextmanager
    def connect(self):
        connection = None
        try:
            if self.postgres:
                try:
                    import psycopg
                    from psycopg.rows import dict_row
                except ImportError as error:
                    raise DatabaseError("PostgreSQL support is not installed.") from error
                connection = psycopg.connect(
                    self.database_url, connect_timeout=8, row_factory=dict_row
                )
            else:
                prefix = "sqlite:///"
                if not self.database_url.startswith(prefix):
                    raise DatabaseError("DATABASE_URL must use PostgreSQL or SQLite.")
                path = self.database_url[len(prefix):]
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                connection = sqlite3.connect(path)
                connection.row_factory = sqlite3.Row
            yield connection
            connection.commit()
        except DatabaseError:
            if connection:
                connection.rollback()
            raise
        except Exception as error:
            if connection:
                connection.rollback()
            raise DatabaseError("MindHabits could not access persistent storage.") from error
        finally:
            if connection:
                connection.close()

    @property
    def placeholder(self):
        return "%s" if self.postgres else "?"

    def initialize(self, connection):
        if self.postgres:
            spreadsheet_schema = """
                CREATE TABLE IF NOT EXISTS spreadsheets (
                    id BIGSERIAL PRIMARY KEY,
                    visitor_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    columns_json JSONB NOT NULL,
                    rows_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """
            analysis_schema = """
                CREATE TABLE IF NOT EXISTS analysis_results (
                    visitor_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    result_json JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (visitor_id, dataset_id, result_type)
                )
            """
        else:
            spreadsheet_schema = """
                CREATE TABLE IF NOT EXISTS spreadsheets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    columns_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """
            analysis_schema = """
                CREATE TABLE IF NOT EXISTS analysis_results (
                    visitor_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (visitor_id, dataset_id, result_type)
                )
            """
        connection.execute(spreadsheet_schema)
        connection.execute(analysis_schema)
        if not self.postgres:
            self._migrate_legacy_sqlite(connection)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS spreadsheets_visitor_idx "
            "ON spreadsheets (visitor_id, updated_at)"
        )

    def _migrate_legacy_sqlite(self, connection):
        """Upgrade the earlier local-only table without affecting PostgreSQL."""
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(spreadsheets)")
        }
        if "data" in columns:
            records = connection.execute("SELECT * FROM spreadsheets").fetchall()
            connection.execute("DROP TABLE IF EXISTS spreadsheets_migrated")
            connection.execute("""
                CREATE TABLE spreadsheets_migrated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    columns_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for row in records:
                try:
                    data = json.loads(row["data"])
                except (TypeError, json.JSONDecodeError):
                    data = {"columns": [], "rows": []}
                visitor_id = row["visitor_id"] if "visitor_id" in columns else ""
                updated_at = row["updated_at"] or datetime.now().isoformat()
                created_at = (
                    row["created_at"] if "created_at" in columns and row["created_at"]
                    else updated_at
                )
                connection.execute(
                    "INSERT INTO spreadsheets_migrated "
                    "(id, visitor_id, name, columns_json, rows_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row["id"], visitor_id, row["name"],
                     json.dumps(data.get("columns", []), ensure_ascii=False),
                     json.dumps(data.get("rows", []), ensure_ascii=False),
                     created_at, updated_at),
                )
            connection.execute("DROP TABLE spreadsheets")
            connection.execute("ALTER TABLE spreadsheets_migrated RENAME TO spreadsheets")
            return
        additions = {
            "visitor_id": "TEXT NOT NULL DEFAULT ''",
            "columns_json": "TEXT",
            "rows_json": "TEXT",
            "created_at": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE spreadsheets ADD COLUMN {name} {definition}"
                )

    def _json_parameter(self, value):
        safe = json_safe(value)
        if self.postgres:
            from psycopg.types.json import Jsonb
            return Jsonb(safe)
        return json.dumps(safe, ensure_ascii=False)

    @staticmethod
    def _decode_json(value):
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise DatabaseError("Stored MindHabits data is unreadable.") from error

    def create_spreadsheet(self, visitor_id, name, columns, rows):
        with self.connect() as connection:
            self.initialize(connection)
            p = self.placeholder
            parameters = (
                visitor_id, name, self._json_parameter(columns), self._json_parameter(rows)
            )
            if self.postgres:
                record = connection.execute(
                    f"INSERT INTO spreadsheets (visitor_id, name, columns_json, rows_json) "
                    f"VALUES ({p}, {p}, {p}, {p}) RETURNING id", parameters
                ).fetchone()
                return record["id"]
            cursor = connection.execute(
                f"INSERT INTO spreadsheets (visitor_id, name, columns_json, rows_json) "
                f"VALUES ({p}, {p}, {p}, {p})", parameters
            )
            return cursor.lastrowid

    def update_spreadsheet(self, visitor_id, spreadsheet_id, name, columns, rows):
        with self.connect() as connection:
            self.initialize(connection)
            p = self.placeholder
            cursor = connection.execute(
                f"UPDATE spreadsheets SET name = {p}, columns_json = {p}, "
                f"rows_json = {p}, updated_at = CURRENT_TIMESTAMP "
                f"WHERE id = {p} AND visitor_id = {p}",
                (name, self._json_parameter(columns), self._json_parameter(rows),
                 spreadsheet_id, visitor_id),
            )
            return cursor.rowcount > 0

    def get_spreadsheet(self, visitor_id, spreadsheet_id):
        with self.connect() as connection:
            self.initialize(connection)
            p = self.placeholder
            record = connection.execute(
                f"SELECT id, name, columns_json, rows_json FROM spreadsheets "
                f"WHERE id = {p} AND visitor_id = {p}",
                (spreadsheet_id, visitor_id),
            ).fetchone()
        if not record:
            return None
        return {
            "id": record["id"], "name": record["name"],
            "columns": self._decode_json(record["columns_json"]),
            "rows": self._decode_json(record["rows_json"]),
        }

    def list_spreadsheets(self, visitor_id):
        with self.connect() as connection:
            self.initialize(connection)
            p = self.placeholder
            if not self.postgres:
                # The first local visitor claims rows from the pre-session schema.
                connection.execute(
                    "UPDATE spreadsheets SET visitor_id = ? WHERE visitor_id = ''",
                    (visitor_id,),
                )
            records = connection.execute(
                f"SELECT id, name FROM spreadsheets WHERE visitor_id = {p} "
                "ORDER BY updated_at DESC, id DESC", (visitor_id,)
            ).fetchall()
        return [{"id": row["id"], "name": row["name"]} for row in records]

    def delete_spreadsheet(self, visitor_id, spreadsheet_id):
        with self.connect() as connection:
            self.initialize(connection)
            p = self.placeholder
            cursor = connection.execute(
                f"DELETE FROM spreadsheets WHERE id = {p} AND visitor_id = {p}",
                (spreadsheet_id, visitor_id),
            )
            return cursor.rowcount > 0

    def save_analysis_result(self, visitor_id, dataset_id, result_type, result):
        if result_type not in {"wellbeing", "habits"}:
            raise DatabaseError("Unsupported analysis result type.")
        with self.connect() as connection:
            self.initialize(connection)
            p = self.placeholder
            value = self._json_parameter(result)
            connection.execute(
                f"INSERT INTO analysis_results "
                f"(visitor_id, dataset_id, result_type, result_json) "
                f"VALUES ({p}, {p}, {p}, {p}) "
                "ON CONFLICT (visitor_id, dataset_id, result_type) DO UPDATE SET "
                "result_json = excluded.result_json, updated_at = CURRENT_TIMESTAMP",
                (visitor_id, dataset_id, result_type, value),
            )

    def get_analysis_results(self, visitor_id, dataset_id):
        if not dataset_id:
            return {"wellbeing": None, "habits": None}
        with self.connect() as connection:
            self.initialize(connection)
            p = self.placeholder
            records = connection.execute(
                f"SELECT result_type, result_json FROM analysis_results "
                f"WHERE visitor_id = {p} AND dataset_id = {p}",
                (visitor_id, dataset_id),
            ).fetchall()
        results = {"wellbeing": None, "habits": None}
        for record in records:
            if record["result_type"] in results:
                results[record["result_type"]] = self._decode_json(record["result_json"])
        return results
