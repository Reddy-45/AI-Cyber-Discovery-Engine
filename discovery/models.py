"""
discovery/models.py — Canonical data model for the Input Processing Layer.

Every input that enters the AI Cyber Discovery Engine — regardless of its
original format (plain text, file upload, JSON, STIX bundle) — is normalized
into a single DiscoveryRequest object before any downstream processing begins.

This is the data contract between the API layer and all pipeline stages.
No pipeline stage inspects raw user input; they only consume DiscoveryRequest.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Input Type Taxonomy ───────────────────────────────────────────────

class InputType(str, Enum):
    """All input types the Discovery Engine understands."""

    # IOC subtypes
    IP_ADDRESS    = "ip_address"    # 203.0.113.42, ::1
    DOMAIN        = "domain"        # malware-c2.ru, api.example.com
    URL           = "url"           # https://evil.com/payload.exe
    FILE_HASH     = "file_hash"     # MD5 / SHA1 / SHA256 / SHA512

    # Structured identifiers
    CVE_ID           = "cve_id"           # CVE-2021-44228
    MALWARE_NAME     = "malware_name"     # WannaCry, Emotet, Cobalt Strike
    APT_GROUP        = "apt_group"        # APT28, Lazarus Group, FIN7
    MITRE_TECHNIQUE  = "mitre_technique"  # T1059, T1059.001, TA0001, G0007

    # Free-form / document inputs
    NATURAL_LANGUAGE = "natural_language" # Incident description, analyst query
    JSON_FILE        = "json_file"        # Uploaded JSON log / event export
    STIX_BUNDLE      = "stix_bundle"      # STIX 2.0 / 2.1 bundle
    CYBER_REPORT     = "cyber_report"     # Plain-text or Markdown threat report


class HashAlgorithm(str, Enum):
    MD5    = "md5"    # 32 hex chars
    SHA1   = "sha1"   # 40 hex chars
    SHA256 = "sha256" # 64 hex chars
    SHA512 = "sha512" # 128 hex chars


class MITREIdType(str, Enum):
    TECHNIQUE = "technique"  # T1234 or T1234.001
    TACTIC    = "tactic"     # TA0001
    GROUP     = "group"      # G0007
    SOFTWARE  = "software"   # S0002


class IPVersion(str, Enum):
    V4 = "ipv4"
    V6 = "ipv6"


class InputSource(str, Enum):
    TEXT_QUERY  = "text_query"   # Plain string from API body / query param
    FILE_UPLOAD = "file_upload"  # Multipart file upload
    API_DIRECT  = "api_direct"   # Direct structured JSON payload


# ── Canonical Discovery Request ────────────────────────────────────────

class DiscoveryRequest(BaseModel):
    """
    The canonical representation of any analyst input.

    Produced by discovery/normalizer.py from raw user input.
    All downstream pipeline stages consume this — never the raw input.

    Design notes:
    - raw_value: exactly what the user provided, preserved for audit
    - normalized_value: cleaned, lowercased, or standardized form
    - metadata: type-specific structured facts extracted during normalization
    - is_valid: False means validation failed but object is still created
      (validation errors are listed in validation_errors)
    """

    request_id:       str       = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at:       datetime  = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Core classification
    input_type:       InputType
    raw_value:        str       = Field(description="Exact original input, unmodified")
    normalized_value: str       = Field(description="Cleaned/standardized form")

    # Provenance
    source:           InputSource = InputSource.TEXT_QUERY
    filename:         str | None = None   # Set when input came from a file upload

    # Validation
    is_valid:         bool       = True
    validation_errors: list[str] = Field(default_factory=list)

    # Type-specific structured metadata extracted during normalization.
    # Examples by type:
    #   ip_address   → {version: "ipv4", is_private: false, is_loopback: false}
    #   file_hash    → {algorithm: "sha256", length: 64}
    #   cve_id       → {year: 2021, sequence: "44228"}
    #   mitre_technique → {id_type: "technique", has_subtechnique: true, parent_id: "T1059"}
    #   url          → {scheme: "https", host: "evil.com", path: "/payload.exe"}
    #   stix_bundle  → {spec_version: "2.1", object_count: 12}
    #   json_file    → {record_count: 50, detected_as: "log_file"}
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Pipeline routing hint — tells downstream stages which processing path to activate
    # Populated by the normalizer; can be overridden by the API caller
    routing_hint: str = ""

    model_config = {"frozen": False}  # Mutable — downstream stages may annotate it


# ── Validation Error Model ─────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Returned alongside a DiscoveryRequest when validation fails."""
    is_valid:  bool
    errors:    list[str] = Field(default_factory=list)
    warnings:  list[str] = Field(default_factory=list)
