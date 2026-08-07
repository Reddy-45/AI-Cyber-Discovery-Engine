"""
discovery/normalizer.py — Input Detection, Validation, and Normalization.

This is the entry point of the AI Cyber Discovery Engine.

Every input — whether typed by a SOC analyst, uploaded as a file, or
submitted via the API — passes through this module before anything else.

Responsibilities:
    1. Detect input type automatically using pattern matching + heuristics
    2. Validate that the input is well-formed for its detected type
    3. Normalize to a canonical string representation
    4. Extract type-specific metadata
    5. Return a fully-populated DiscoveryRequest

What this module does NOT do:
    - No database lookups
    - No network calls
    - No embedding generation
    - No semantic search

Detection Priority Order (most-specific first):
    URL → CVE ID → MITRE ID → IP Address → File Hash →
    APT Group → Domain → Malware Name → Natural Language

File inputs follow a separate path:
    STIX bundle → JSON log file → Cyber report (plain text)
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from discovery.models import (
    DiscoveryRequest,
    HashAlgorithm,
    InputSource,
    InputType,
    IPVersion,
    MITREIdType,
    ValidationResult,
)


# ── Compiled Patterns ──────────────────────────────────────────────────
# Pre-compile all regexes at module load time.

_RE_CVE         = re.compile(r"^CVE-\d{4}-\d{4,7}$", re.IGNORECASE)
_RE_MITRE_TECH  = re.compile(r"^T\d{4}(\.\d{3})?$", re.IGNORECASE)
_RE_MITRE_TAC   = re.compile(r"^TA\d{4}$", re.IGNORECASE)
_RE_MITRE_GROUP = re.compile(r"^G\d{4}$", re.IGNORECASE)
_RE_MITRE_SW    = re.compile(r"^S\d{4}$", re.IGNORECASE)
_RE_MD5         = re.compile(r"^[0-9a-fA-F]{32}$")
_RE_SHA1        = re.compile(r"^[0-9a-fA-F]{40}$")
_RE_SHA256      = re.compile(r"^[0-9a-fA-F]{64}$")
_RE_SHA512      = re.compile(r"^[0-9a-fA-F]{128}$")
_RE_APT_NUM     = re.compile(r"^APT\d+$", re.IGNORECASE)
_RE_FIN_NUM     = re.compile(r"^FIN\d+$", re.IGNORECASE)
_RE_TA_NUM      = re.compile(r"^TA\d+$", re.IGNORECASE)   # threat actor group names
_RE_DOMAIN      = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)"
    r"+[a-zA-Z]{2,}$"
)

# Known APT group aliases not captured by numbering patterns
_KNOWN_APT_ALIASES: frozenset[str] = frozenset({
    "lazarus group", "lazarus", "hidden cobra",
    "equation group",
    "fancy bear", "sofacy",
    "cozy bear", "the dukes",
    "comment crew",
    "aurora panda",
    "deep panda",
    "emissary panda",
    "wocao",
    "wizard spider",
    "sandworm",
    "turla",
    "carbanak",
    "fin6", "fin7", "fin8",
    "ta505", "ta542", "ta544",
    "charming kitten",
    "phosphorus",
    "muddy water",
    "muddywater",
    "silence group",
    "evil corp",
})

# Known malware family names (lowercase)
_KNOWN_MALWARE: frozenset[str] = frozenset({
    "wannacry", "wannacrypt", "wanacrypt0r", "wcry",
    "emotet", "geodo", "heodo",
    "trickbot", "trickloader",
    "ryuk",
    "cobalt strike",
    "mimikatz",
    "metasploit",
    "empire", "powershell empire",
    "cobaltstrike",
    "lockbit", "lockbit 2.0", "lockbit 3.0",
    "revil", "sodinokibi",
    "conti",
    "qakbot", "qbot",
    "icedid",
    "bazarloader", "bazarbackdoor",
    "dridex",
    "njrat", "njw0rm",
    "remcos",
    "asyncrat",
    "nanocore",
    "netwire",
    "gh0st rat",
    "poison ivy",
    "darkcomet",
    "blackmatter",
    "darkside",
    "hive ransomware",
    "alphv", "blackcat",
})


# ── Public API ─────────────────────────────────────────────────────────

def normalize_text(raw: str, source: InputSource = InputSource.TEXT_QUERY) -> DiscoveryRequest:
    """
    Normalize a text query submitted by an analyst.

    Detects type, validates, extracts metadata, and returns a
    DiscoveryRequest. This is the primary entry point for plain-text input.

    Args:
        raw:    The raw string exactly as the analyst typed or pasted it.
        source: Where the input came from (default: text query).

    Returns:
        A fully-populated DiscoveryRequest (is_valid may be False).
    """
    value = raw.strip()
    if not value:
        return _make_invalid(raw, InputType.NATURAL_LANGUAGE, "Input is empty", source)

    # Detection pipeline — ordered from most specific to least
    detectors = [
        _detect_url,
        _detect_cve,
        _detect_mitre,
        _detect_ip,
        _detect_hash,
        _detect_apt_group,
        _detect_domain,
        _detect_malware,
    ]

    for detector in detectors:
        result = detector(value)
        if result is not None:
            result.source = source
            return result

    # Fallback: natural language
    return _detect_natural_language(value, source)


def normalize_file(
    content: bytes,
    filename: str,
    source: InputSource = InputSource.FILE_UPLOAD,
) -> DiscoveryRequest:
    """
    Normalize a file uploaded by an analyst.

    Detects whether the file is a STIX bundle, a JSON log, or a plain-text
    cyber report. Returns a DiscoveryRequest with the full file content in
    normalized_value and structural metadata extracted.

    Args:
        content:  Raw bytes of the uploaded file.
        filename: Original filename (used for extension hinting).
        source:   Input source tag.

    Returns:
        A fully-populated DiscoveryRequest.
    """
    if not content:
        return _make_invalid(filename, InputType.CYBER_REPORT, "Uploaded file is empty", source,
                             filename=filename)

    ext = Path(filename).suffix.lower() if filename else ""

    # Try to parse as JSON first (covers STIX and JSON log files)
    if ext in (".json", "") or content.lstrip()[:1] == b"{":
        try:
            parsed = json.loads(content.decode("utf-8", errors="replace"))
            return _detect_json_content(parsed, content, filename, source)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # Not JSON — treat as text report

    # Plain-text / markdown / PDF (text extraction only — no PDF lib needed here)
    return _make_cyber_report(content, filename, source)


# ── Type Detectors ─────────────────────────────────────────────────────
# Each returns DiscoveryRequest | None.
# None means "not this type; try the next detector".

def _detect_url(value: str) -> DiscoveryRequest | None:
    """Detect if value is a URL (has a recognized protocol prefix)."""
    if not re.match(r"^(https?|ftp|ftps)://", value, re.IGNORECASE):
        return None

    errors: list[str] = []
    metadata: dict[str, Any] = {}

    try:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        metadata = {
            "scheme": parsed.scheme.lower(),
            "host": host,
            "path": parsed.path or "/",
            "query": parsed.query or "",
            "port": parsed.port,
            "fragment": parsed.fragment or "",
        }
        if not host:
            errors.append("URL has no host component")
    except Exception as exc:
        errors.append(f"URL parse error: {exc}")

    return DiscoveryRequest(
        input_type=InputType.URL,
        raw_value=value,
        normalized_value=value.lower().rstrip("/"),
        metadata=metadata,
        is_valid=len(errors) == 0,
        validation_errors=errors,
        routing_hint="ioc_url",
    )


def _detect_cve(value: str) -> DiscoveryRequest | None:
    """Detect CVE ID: CVE-YYYY-NNNNN (4-7 digit sequence)."""
    if not _RE_CVE.match(value):
        return None

    upper = value.upper()
    parts = upper.split("-")   # ["CVE", "2021", "44228"]
    metadata = {
        "year": int(parts[1]),
        "sequence": parts[2],
    }
    return DiscoveryRequest(
        input_type=InputType.CVE_ID,
        raw_value=value,
        normalized_value=upper,
        metadata=metadata,
        routing_hint="cve_lookup",
    )


def _detect_mitre(value: str) -> DiscoveryRequest | None:
    """Detect MITRE ATT&CK IDs: T####, T####.###, TA####, G####, S####."""
    v = value.upper()

    if _RE_MITRE_TAC.match(v):
        id_type = MITREIdType.TACTIC
        routing = "mitre_tactic"
    elif _RE_MITRE_GROUP.match(v):
        id_type = MITREIdType.GROUP
        routing = "mitre_group"
    elif _RE_MITRE_SW.match(v):
        id_type = MITREIdType.SOFTWARE
        routing = "mitre_software"
    elif _RE_MITRE_TECH.match(v):
        id_type = MITREIdType.TECHNIQUE
        routing = "mitre_technique"
    else:
        return None

    has_sub = "." in v
    parent_id = v.split(".")[0] if has_sub else None

    return DiscoveryRequest(
        input_type=InputType.MITRE_TECHNIQUE,
        raw_value=value,
        normalized_value=v,
        metadata={
            "id_type": id_type.value,
            "has_subtechnique": has_sub,
            "parent_id": parent_id,
        },
        routing_hint=routing,
    )


def _detect_ip(value: str) -> DiscoveryRequest | None:
    """Detect IPv4 and IPv6 addresses using Python's ipaddress stdlib."""
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return None

    version = IPVersion.V4 if addr.version == 4 else IPVersion.V6
    errors: list[str] = []

    metadata = {
        "version": version.value,
        "is_private": addr.is_private,
        "is_loopback": addr.is_loopback,
        "is_multicast": addr.is_multicast,
        "is_global": addr.is_global,
        "compressed": str(addr),
    }

    return DiscoveryRequest(
        input_type=InputType.IP_ADDRESS,
        raw_value=value,
        normalized_value=str(addr),  # compressed canonical form
        metadata=metadata,
        is_valid=True,
        routing_hint="ioc_ip",
    )


def _detect_hash(value: str) -> DiscoveryRequest | None:
    """Detect cryptographic file hashes by length and hex character set."""
    if _RE_MD5.match(value):
        algo, length = HashAlgorithm.MD5, 32
    elif _RE_SHA1.match(value):
        algo, length = HashAlgorithm.SHA1, 40
    elif _RE_SHA256.match(value):
        algo, length = HashAlgorithm.SHA256, 64
    elif _RE_SHA512.match(value):
        algo, length = HashAlgorithm.SHA512, 128
    else:
        return None

    return DiscoveryRequest(
        input_type=InputType.FILE_HASH,
        raw_value=value,
        normalized_value=value.lower(),
        metadata={"algorithm": algo.value, "length": length},
        routing_hint="ioc_hash",
    )


def _detect_apt_group(value: str) -> DiscoveryRequest | None:
    """Detect APT group designations by numbered pattern or known alias."""
    lower = value.lower().strip()
    upper = value.upper().strip()

    is_numbered = (
        _RE_APT_NUM.match(upper)
        or _RE_FIN_NUM.match(upper)
        or _RE_TA_NUM.match(upper)
    )
    is_known_alias = lower in _KNOWN_APT_ALIASES

    if not (is_numbered or is_known_alias):
        return None

    # Normalize: "apt28" → "APT28", "fancy bear" → "Fancy Bear"
    if is_numbered:
        normalized = upper
    else:
        normalized = value.strip().title()

    return DiscoveryRequest(
        input_type=InputType.APT_GROUP,
        raw_value=value,
        normalized_value=normalized,
        metadata={
            "pattern_type": "numbered" if is_numbered else "named",
            "is_known_alias": is_known_alias,
        },
        routing_hint="apt_lookup",
    )


def _detect_domain(value: str) -> DiscoveryRequest | None:
    """Detect FQDNs (hostname.tld) that are not URLs."""
    if not _RE_DOMAIN.match(value):
        return None

    # Additional heuristic: reject values that look like sentences
    if " " in value:
        return None

    parts = value.lower().split(".")
    tld = "." + parts[-1]
    apex = ".".join(parts[-2:]) if len(parts) >= 2 else value.lower()
    subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""

    return DiscoveryRequest(
        input_type=InputType.DOMAIN,
        raw_value=value,
        normalized_value=value.lower(),
        metadata={
            "tld": tld,
            "apex_domain": apex,
            "subdomain": subdomain,
            "label_count": len(parts),
        },
        routing_hint="ioc_domain",
    )


def _detect_malware(value: str) -> DiscoveryRequest | None:
    """Detect malware family names by matching against the known-malware set."""
    lower = value.lower().strip()
    if lower not in _KNOWN_MALWARE:
        return None

    return DiscoveryRequest(
        input_type=InputType.MALWARE_NAME,
        raw_value=value,
        normalized_value=value.strip().title(),
        metadata={"matched_canonical": lower},
        routing_hint="malware_lookup",
    )


def _detect_natural_language(
    value: str, source: InputSource
) -> DiscoveryRequest:
    """
    Fallback: treat input as a natural-language incident description.
    This is always valid — the analyst is free to describe anything.
    """
    words = value.split()
    return DiscoveryRequest(
        input_type=InputType.NATURAL_LANGUAGE,
        raw_value=value,
        normalized_value=value.strip(),
        source=source,
        metadata={
            "word_count": len(words),
            "char_count": len(value),
        },
        routing_hint="nlp_query",
    )


# ── File Content Detectors ─────────────────────────────────────────────

def _detect_json_content(
    parsed: Any,
    raw_bytes: bytes,
    filename: str,
    source: InputSource,
) -> DiscoveryRequest:
    """Classify a successfully-parsed JSON file as STIX bundle or JSON log."""
    errors: list[str] = []

    # STIX 2.x bundle detection
    if (
        isinstance(parsed, dict)
        and parsed.get("type") == "bundle"
        and "objects" in parsed
    ):
        objects = parsed.get("objects", [])
        spec_version = parsed.get("spec_version", "unknown")

        # Count STIX object types
        type_counts: dict[str, int] = {}
        for obj in objects:
            ot = obj.get("type", "unknown") if isinstance(obj, dict) else "unknown"
            type_counts[ot] = type_counts.get(ot, 0) + 1

        return DiscoveryRequest(
            input_type=InputType.STIX_BUNDLE,
            raw_value=filename,
            normalized_value=raw_bytes.decode("utf-8", errors="replace"),
            source=source,
            filename=filename,
            metadata={
                "spec_version": spec_version,
                "object_count": len(objects),
                "object_types": type_counts,
                "has_indicators": "indicator" in type_counts,
                "has_relationships": "relationship" in type_counts,
                "has_malware": "malware" in type_counts,
            },
            routing_hint="stix_ingest",
        )

    # JSON log file detection (list of dicts or single dict with event fields)
    if isinstance(parsed, list) and len(parsed) > 0:
        record_count = len(parsed)
        sample = parsed[0] if isinstance(parsed[0], dict) else {}
        # Check if it looks like a log event (common field names)
        log_fields = {"timestamp", "source_ip", "event_type", "severity",
                      "src_ip", "dst_ip", "message", "level", "time"}
        detected_as = "log_file" if log_fields & set(sample.keys()) else "generic_json"
        return DiscoveryRequest(
            input_type=InputType.JSON_FILE,
            raw_value=filename,
            normalized_value=raw_bytes.decode("utf-8", errors="replace"),
            source=source,
            filename=filename,
            metadata={
                "record_count": record_count,
                "detected_as": detected_as,
                "sample_fields": list(sample.keys())[:10],
            },
            routing_hint="json_log_ingest" if detected_as == "log_file" else "json_generic",
        )

    if isinstance(parsed, dict):
        return DiscoveryRequest(
            input_type=InputType.JSON_FILE,
            raw_value=filename,
            normalized_value=raw_bytes.decode("utf-8", errors="replace"),
            source=source,
            filename=filename,
            metadata={"record_count": 1, "detected_as": "single_object",
                      "keys": list(parsed.keys())[:10]},
            routing_hint="json_generic",
        )

    # JSON but not a recognized structure
    return DiscoveryRequest(
        input_type=InputType.JSON_FILE,
        raw_value=filename,
        normalized_value=raw_bytes.decode("utf-8", errors="replace"),
        source=source,
        filename=filename,
        metadata={"detected_as": "unrecognised_json"},
        routing_hint="json_generic",
    )


def _make_cyber_report(
    content: bytes,
    filename: str,
    source: InputSource,
) -> DiscoveryRequest:
    """Treat content as a plain-text cyber threat report."""
    ext = Path(filename).suffix.lower() if filename else ""
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    return DiscoveryRequest(
        input_type=InputType.CYBER_REPORT,
        raw_value=filename,
        normalized_value=text,
        source=source,
        filename=filename,
        metadata={
            "file_type": ext.lstrip(".") or "unknown",
            "char_count": len(text),
            "line_count": text.count("\n"),
        },
        routing_hint="report_ingest",
    )


# ── Helpers ────────────────────────────────────────────────────────────

def _make_invalid(
    raw: str,
    input_type: InputType,
    error: str,
    source: InputSource,
    filename: str | None = None,
) -> DiscoveryRequest:
    """Construct an invalid DiscoveryRequest with a single error message."""
    return DiscoveryRequest(
        input_type=input_type,
        raw_value=raw,
        normalized_value=raw,
        source=source,
        filename=filename,
        is_valid=False,
        validation_errors=[error],
        routing_hint="invalid",
    )


def validate(request: DiscoveryRequest) -> ValidationResult:
    """
    Run additional validation checks on an already-normalized request.

    Used by the API layer to surface warnings (not just errors) to callers.
    Currently checks for private IPs, suspicious TLDs, and short NL queries.
    """
    errors = list(request.validation_errors)
    warnings: list[str] = []

    if request.input_type == InputType.IP_ADDRESS:
        if request.metadata.get("is_private"):
            warnings.append("IP address is in a private range — threat intel lookups may not return results")
        if request.metadata.get("is_loopback"):
            warnings.append("IP address is a loopback address (127.0.0.1 / ::1)")

    elif request.input_type == InputType.NATURAL_LANGUAGE:
        word_count = request.metadata.get("word_count", 0)
        if word_count < 3:
            warnings.append(
                "Query is very short. For better results, describe the incident in more detail."
            )

    elif request.input_type == InputType.DOMAIN:
        suspicious_tlds = {".ru", ".cn", ".tk", ".xyz", ".top", ".pw", ".cc", ".su"}
        tld = request.metadata.get("tld", "")
        if tld in suspicious_tlds:
            warnings.append(f"Domain uses TLD '{tld}' which is commonly associated with malicious infrastructure")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
