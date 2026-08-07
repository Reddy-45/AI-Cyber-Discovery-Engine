"""
engine/ingest.py — Pipeline Stage ① + ②: Ingestion and Normalization.

Loads raw JSON log files from disk and transforms them into
CanonicalEvent objects — the uniform data type used by every
downstream stage.

Why combined? Ingestion and normalization always run together.
Separating them into distinct files adds a hand-off with no benefit
at this scale. A single function per log source is clean enough.

Supported sources:
    - auth_logs.json  (authentication events)
    - firewall_logs.json  (network/firewall events)
    - network_logs.json  (network and malware events)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from engine.models import CanonicalEvent, EventType, Severity

logger = logging.getLogger(__name__)

# Severity string → Severity enum (handles any capitalisation)
_SEVERITY_MAP: dict[str, Severity] = {s.value: s for s in Severity}

# EventType string → EventType enum
_EVENT_TYPE_MAP: dict[str, EventType] = {e.value: e for e in EventType}


def _parse_timestamp(raw: str) -> datetime:
    """Parse ISO 8601 timestamp string to datetime. Returns UTC-aware."""
    raw = raw.strip()
    # Try ISO formats
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Fallback: fromisoformat
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        logger.warning("Could not parse timestamp '%s', using now()", raw)
        return datetime.now(tz=timezone.utc)


def _normalize_event(raw: dict) -> CanonicalEvent:
    """
    Transform a raw log dict into a CanonicalEvent.

    Missing optional fields default gracefully — no exceptions
    for absent fields.
    """
    severity_str = str(raw.get("severity", "info")).lower()
    event_type_str = str(raw.get("event_type", "unknown")).lower()

    return CanonicalEvent(
        timestamp=_parse_timestamp(raw.get("timestamp", "")),
        source_ip=raw.get("source_ip"),
        dest_ip=raw.get("dest_ip"),
        source_port=raw.get("source_port"),
        dest_port=raw.get("dest_port"),
        protocol=raw.get("protocol"),
        event_type=_EVENT_TYPE_MAP.get(event_type_str, EventType.UNKNOWN),
        severity=_SEVERITY_MAP.get(severity_str, Severity.INFO),
        description=raw.get("description", ""),
        raw=json.dumps(raw),
        metadata={k: v for k, v in raw.items()
                  if k not in {"timestamp", "source_ip", "dest_ip", "source_port",
                               "dest_port", "protocol", "event_type", "severity", "description"}},
    )


def load_and_normalize(log_dir: str | Path) -> list[CanonicalEvent]:
    """
    Load all JSON log files from a directory and return normalized events.

    Reads auth_logs.json, firewall_logs.json, and network_logs.json.
    Unknown files are skipped with a warning. Malformed records are
    skipped individually so one bad line doesn't abort the pipeline.

    Args:
        log_dir: Path to the directory containing JSON log files.

    Returns:
        List of CanonicalEvent objects, sorted by timestamp ascending.
    """
    log_path = Path(log_dir)
    if not log_path.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_path}")

    events: list[CanonicalEvent] = []

    for json_file in sorted(log_path.glob("*.json")):
        logger.info("Loading %s", json_file.name)
        try:
            raw_records: list[dict] = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read %s: %s", json_file.name, exc)
            continue

        for record in raw_records:
            try:
                events.append(_normalize_event(record))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed record in %s: %s", json_file.name, exc)

    # Sort chronologically so correlation works correctly
    events.sort(key=lambda e: e.timestamp)
    logger.info("Loaded and normalized %d events from %s", len(events), log_path)
    return events
