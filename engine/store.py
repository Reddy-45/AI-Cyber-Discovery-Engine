"""
engine/store.py — Pipeline Stage ⑤: SQLite Persistence.

Stores analysis results so the dashboard can query them without
re-running the pipeline on every page refresh.

Deliberately simple:
    - Raw SQL — no ORM
    - Three tables: events, alerts, mitre_techniques
    - Five functions: init_db, save_results, load_results, clear_db, get_stats
    - Schema created with CREATE TABLE IF NOT EXISTS — no migrations needed
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from engine.config import load_config
from engine.models import (
    AnalysisResult,
    EnrichedEvent,
    MITRETechnique,
    RiskScore,
    Severity,
    ThreatAlert,
)

logger = logging.getLogger(__name__)


# ── Connection helper ─────────────────────────────────────────────────

@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Open a database connection and close it when done."""
    cfg = load_config()
    db_path = Path(cfg["data"]["database_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    title           TEXT NOT NULL,
    threat_type     TEXT NOT NULL,
    severity        TEXT NOT NULL,
    risk_total      REAL NOT NULL,
    risk_severity   REAL DEFAULT 0,
    risk_anomaly    REAL DEFAULT 0,
    risk_mitre      REAL DEFAULT 0,
    risk_frequency  REAL DEFAULT 0,
    risk_asset      REAL DEFAULT 0,
    anomaly_score   REAL DEFAULT 0,
    explanation     TEXT DEFAULT '',
    mitigation      TEXT DEFAULT '[]',
    source_ips      TEXT DEFAULT '[]',
    dest_ips        TEXT DEFAULT '[]',
    event_ids       TEXT DEFAULT '[]',
    iocs            TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS mitre_techniques (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id        TEXT NOT NULL REFERENCES alerts(alert_id),
    technique_id    TEXT NOT NULL,
    technique_name  TEXT NOT NULL,
    tactic          TEXT NOT NULL,
    tactic_order    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_log (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at          TEXT NOT NULL,
    total_events    INTEGER DEFAULT 0,
    total_alerts    INTEGER DEFAULT 0,
    high_risk_count INTEGER DEFAULT 0
);
"""


# ── Public API ────────────────────────────────────────────────────────

def init_db() -> None:
    """Create database tables if they don't already exist."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("Database initialized")


def save_results(result: AnalysisResult) -> None:
    """
    Persist an AnalysisResult to the database.

    Clears previous results before inserting — this is a demo engine,
    not a historical SIEM. Each run replaces the previous state.
    """
    with _connect() as conn:
        # Clear previous run
        conn.execute("DELETE FROM mitre_techniques")
        conn.execute("DELETE FROM alerts")

        # Insert new alerts
        for alert in result.alerts:
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, created_at, title, threat_type, severity,
                    risk_total, risk_severity, risk_anomaly, risk_mitre,
                    risk_frequency, risk_asset, anomaly_score,
                    explanation, mitigation, source_ips, dest_ips,
                    event_ids, iocs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.created_at.isoformat(),
                    alert.title,
                    alert.threat_type,
                    alert.severity.value,
                    alert.risk_score.total,
                    alert.risk_score.severity_component,
                    alert.risk_score.anomaly_component,
                    alert.risk_score.mitre_stage_component,
                    alert.risk_score.frequency_component,
                    alert.risk_score.asset_criticality_component,
                    alert.anomaly_score,
                    alert.explanation,
                    json.dumps(alert.mitigation),
                    json.dumps(alert.source_ips),
                    json.dumps(alert.dest_ips),
                    json.dumps(alert.event_ids),
                    json.dumps([ioc.model_dump() for ioc in alert.iocs]),
                ),
            )

            # Insert MITRE techniques for this alert
            for tech in alert.mitre_techniques:
                conn.execute(
                    """
                    INSERT INTO mitre_techniques
                        (alert_id, technique_id, technique_name, tactic, tactic_order)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (alert.alert_id, tech.technique_id, tech.technique_name,
                     tech.tactic, tech.tactic_order),
                )

        # Log the run
        conn.execute(
            "INSERT INTO run_log (ran_at, total_events, total_alerts, high_risk_count) VALUES (?, ?, ?, ?)",
            (result.analysed_at.isoformat(), result.total_events,
             len(result.alerts), result.high_risk_count),
        )

    logger.info("Saved %d alerts to database", len(result.alerts))


def load_results() -> AnalysisResult | None:
    """
    Load the most recent analysis results from the database.

    Returns None if no results have been saved yet.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY risk_total DESC"
        ).fetchall()

        if not rows:
            return None

        alerts: list[ThreatAlert] = []
        for row in rows:
            # Load MITRE techniques for this alert
            tech_rows = conn.execute(
                "SELECT * FROM mitre_techniques WHERE alert_id = ? ORDER BY tactic_order",
                (row["alert_id"],),
            ).fetchall()

            techniques = [
                MITRETechnique(
                    technique_id=t["technique_id"],
                    technique_name=t["technique_name"],
                    tactic=t["tactic"],
                    tactic_order=t["tactic_order"],
                )
                for t in tech_rows
            ]

            risk = RiskScore(
                total=row["risk_total"],
                severity_component=row["risk_severity"],
                anomaly_component=row["risk_anomaly"],
                mitre_stage_component=row["risk_mitre"],
                frequency_component=row["risk_frequency"],
                asset_criticality_component=row["risk_asset"],
            )

            alerts.append(ThreatAlert(
                alert_id=row["alert_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                title=row["title"],
                threat_type=row["threat_type"],
                severity=Severity(row["severity"]),
                risk_score=risk,
                anomaly_score=row["anomaly_score"],
                explanation=row["explanation"],
                mitigation=json.loads(row["mitigation"]),
                source_ips=json.loads(row["source_ips"]),
                dest_ips=json.loads(row["dest_ips"]),
                event_ids=json.loads(row["event_ids"]),
                mitre_techniques=techniques,
            ))

        # Get run metadata from the latest run log
        run_row = conn.execute(
            "SELECT * FROM run_log ORDER BY run_id DESC LIMIT 1"
        ).fetchone()

        return AnalysisResult(
            alerts=alerts,
            total_events=run_row["total_events"] if run_row else 0,
            high_risk_count=sum(1 for a in alerts if a.risk_score.total >= 70),
        )


def get_stats() -> dict:
    """Return quick statistics for the dashboard header."""
    with _connect() as conn:
        alert_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        high_risk = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE risk_total >= 70"
        ).fetchone()[0]
        last_run = conn.execute(
            "SELECT ran_at, total_events FROM run_log ORDER BY run_id DESC LIMIT 1"
        ).fetchone()

    return {
        "total_alerts": alert_count,
        "high_risk_alerts": high_risk,
        "last_run": last_run["ran_at"] if last_run else None,
        "total_events_processed": last_run["total_events"] if last_run else 0,
    }


def clear_db() -> None:
    """Wipe all data — useful for resetting between demos."""
    with _connect() as conn:
        conn.execute("DELETE FROM mitre_techniques")
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM run_log")
    logger.info("Database cleared")
