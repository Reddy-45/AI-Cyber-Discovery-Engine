"""
engine/enrich.py — Pipeline Stage ③: Enrichment.

Takes CanonicalEvents and adds threat intelligence context:

    1. IOC Extraction  — pulls IPs, hashes, domains from event fields
    2. Reputation Lookup — checks extracted IOCs against local threat intel
    3. MITRE Mapping   — matches event descriptions to ATT&CK techniques

Why this exists:
    Raw events are data. Enriched events are information.
    "Connection from 203.0.113.42" is a data point.
    "Connection from a KNOWN SCANNER (confidence 0.95)" is actionable.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from engine.config import load_config
from engine.models import (
    CanonicalEvent,
    EnrichedEvent,
    IOCMatch,
    IOCType,
    MITRETechnique,
)

logger = logging.getLogger(__name__)

# ── Regex patterns for IOC extraction ───────────────────────────────

_RE_IPV4   = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_RE_HASH_MD5   = re.compile(r'\b[a-fA-F0-9]{32}\b')
_RE_HASH_SHA256 = re.compile(r'\b[a-fA-F0-9]{64}\b')
_RE_DOMAIN = re.compile(r'\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b', re.I)
_RE_CVE    = re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.I)

# Private IP ranges — excluded from IP-based IOC flagging
_PRIVATE_RANGES = re.compile(
    r'^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|0\.)'
)


def _load_threat_intel() -> dict:
    """Load the local threat intelligence JSON file."""
    cfg = load_config()
    path = Path(cfg["data"]["threat_intel_path"])
    if not path.exists():
        logger.warning("Threat intel file not found: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mitre_db() -> list[dict]:
    """Load the local MITRE ATT&CK technique list."""
    cfg = load_config()
    path = Path(cfg["data"]["mitre_path"])
    if not path.exists():
        logger.warning("MITRE ATT&CK file not found: %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("techniques", [])


# ── IOC Extraction ───────────────────────────────────────────────────

def _extract_iocs(event: CanonicalEvent, threat_intel: dict) -> tuple[list[IOCMatch], float]:
    """
    Extract IOC candidates from event fields and check reputation.

    Returns:
        (ioc_list, reputation_score)
        reputation_score: 0.0 = definitely malicious, 1.0 = definitely clean
    """
    iocs: list[IOCMatch] = []
    bad_ip_map = {entry["ip"]: entry for entry in threat_intel.get("malicious_ips", [])}
    bad_hash_map = {entry["hash"]: entry for entry in threat_intel.get("malicious_hashes", [])}

    searchable = f"{event.description} {event.raw}"
    worst_reputation = 0.5  # unknown by default
    found_bad = False

    # IPs — from structured fields first (most reliable)
    candidate_ips: set[str] = set()
    for ip_field in (event.source_ip, event.dest_ip):
        if ip_field and not _PRIVATE_RANGES.match(ip_field):
            candidate_ips.add(ip_field)

    # IPs — also from description text
    for ip in _RE_IPV4.findall(searchable):
        if not _PRIVATE_RANGES.match(ip):
            candidate_ips.add(ip)

    for ip in candidate_ips:
        is_bad = ip in bad_ip_map
        confidence = bad_ip_map[ip]["confidence"] if is_bad else 0.4
        iocs.append(IOCMatch(ioc_type=IOCType.IP_ADDRESS, value=ip,
                             confidence=confidence, is_known_bad=is_bad))
        if is_bad:
            found_bad = True
            worst_reputation = min(worst_reputation, 1.0 - confidence)

    # MD5 hashes
    for h in _RE_HASH_MD5.findall(searchable):
        is_bad = h in bad_hash_map
        confidence = bad_hash_map[h]["confidence"] if is_bad else 0.3
        iocs.append(IOCMatch(ioc_type=IOCType.FILE_HASH, value=h,
                             confidence=confidence, is_known_bad=is_bad))
        if is_bad:
            found_bad = True
            worst_reputation = min(worst_reputation, 1.0 - confidence)

    # CVEs
    for cve in _RE_CVE.findall(searchable):
        iocs.append(IOCMatch(ioc_type=IOCType.CVE, value=cve.upper(),
                             confidence=0.9, is_known_bad=True))
        found_bad = True

    reputation = worst_reputation if found_bad else 0.6
    return iocs, reputation


# ── MITRE ATT&CK Mapping ─────────────────────────────────────────────

def _map_mitre_techniques(event: CanonicalEvent, mitre_db: list[dict]) -> list[MITRETechnique]:
    """
    Match an event's description against MITRE ATT&CK keyword lists.

    Returns all techniques whose keywords appear in the event description.
    """
    text = event.description.lower()
    matched: list[MITRETechnique] = []

    for technique in mitre_db:
        if any(kw in text for kw in technique.get("keywords", [])):
            matched.append(MITRETechnique(
                technique_id=technique["technique_id"],
                technique_name=technique["technique_name"],
                tactic=technique["tactic"],
                tactic_order=technique.get("tactic_order", 0),
            ))

    return matched


# ── Public API ───────────────────────────────────────────────────────

def enrich_events(events: list[CanonicalEvent]) -> list[EnrichedEvent]:
    """
    Enrich a list of canonical events with IOCs, reputation, and MITRE context.

    Args:
        events: Normalized events from the ingestion stage.

    Returns:
        List of EnrichedEvent objects, one per input event.
    """
    threat_intel = _load_threat_intel()
    mitre_db = _load_mitre_db()
    enriched: list[EnrichedEvent] = []

    for event in events:
        iocs, reputation = _extract_iocs(event, threat_intel)
        mitre_techniques = _map_mitre_techniques(event, mitre_db)

        enriched.append(EnrichedEvent(
            event=event,
            iocs=iocs,
            mitre_techniques=mitre_techniques,
            reputation_score=reputation,
        ))

    logger.info("Enriched %d events", len(enriched))
    return enriched
