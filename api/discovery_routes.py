"""
api/discovery_routes.py — FastAPI endpoints for the Input Processing Layer.

These endpoints accept any supported analyst input, normalize it, validate it,
and return a structured DiscoveryRequest. No pipeline processing yet —
this phase is purely about normalizing inputs into canonical form.

New endpoints:
    POST /api/v1/discover         — text query (JSON body)
    POST /api/v1/discover/upload  — file upload (multipart)
    GET  /api/v1/discover/types   — list all supported input types

Existing Phase 2 endpoints are completely unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from pydantic import BaseModel, Field

from discovery.normalizer import normalize_text, normalize_file, validate
from discovery.models import DiscoveryRequest, InputSource, InputType

router = APIRouter()

# ── Request / Response Schemas ─────────────────────────────────────────

class TextQueryRequest(BaseModel):
    """Body schema for POST /discover."""
    query: str = Field(
        description="Any analyst input: IP, domain, URL, hash, CVE, malware name, "
                    "APT group, MITRE ID, or natural-language description.",
        min_length=1,
        max_length=10_000,
        examples=["203.0.113.42", "CVE-2021-44228", "APT28", "T1059.001",
                  "Suspicious PowerShell execution followed by outbound C2 traffic"],
    )


class DiscoveryRequestOut(BaseModel):
    """API response schema — a serializable view of DiscoveryRequest."""
    request_id:        str
    created_at:        str        # ISO-8601 string
    input_type:        str
    raw_value:         str
    normalized_value:  str
    source:            str
    filename:          str | None
    is_valid:          bool
    validation_errors: list[str]
    warnings:          list[str]
    metadata:          dict
    routing_hint:      str

    @classmethod
    def from_request(
        cls,
        req: DiscoveryRequest,
        warnings: list[str] | None = None,
    ) -> "DiscoveryRequestOut":
        # Truncate large normalized values in the response (file contents)
        norm = req.normalized_value
        if len(norm) > 500:
            norm = norm[:500] + f"... [truncated, {len(req.normalized_value)} chars total]"

        return cls(
            request_id=req.request_id,
            created_at=req.created_at.isoformat(),
            input_type=req.input_type.value,
            raw_value=req.raw_value,
            normalized_value=norm,
            source=req.source.value,
            filename=req.filename,
            is_valid=req.is_valid,
            validation_errors=req.validation_errors,
            warnings=warnings or [],
            metadata=req.metadata,
            routing_hint=req.routing_hint,
        )


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post(
    "/discover",
    response_model=DiscoveryRequestOut,
    summary="Normalize any text-based analyst input",
    description=(
        "Submit any text — IP address, domain, URL, file hash, CVE ID, "
        "malware name, APT group, MITRE technique, or a natural-language "
        "incident description. The engine automatically detects the type, "
        "validates the input, and returns a normalized DiscoveryRequest "
        "ready for the investigation pipeline."
    ),
    responses={
        200: {"description": "Input successfully normalized"},
        400: {"description": "Input is empty or exceeds maximum length"},
    },
)
def discover_text(body: TextQueryRequest) -> DiscoveryRequestOut:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    req = normalize_text(query, source=InputSource.TEXT_QUERY)
    result = validate(req)

    return DiscoveryRequestOut.from_request(req, warnings=result.warnings)


@router.post(
    "/discover/upload",
    response_model=DiscoveryRequestOut,
    summary="Normalize an uploaded file (STIX bundle, JSON logs, or text report)",
    description=(
        "Upload a file for the discovery engine to classify and normalize. "
        "Supported formats:\n"
        "- **STIX 2.x bundle** (`.json` with `type: 'bundle'`)\n"
        "- **JSON log file** (array of event objects)\n"
        "- **Plain-text threat report** (`.txt`, `.md`)\n\n"
        "The engine auto-detects the format from file content, not the extension."
    ),
    responses={
        200: {"description": "File successfully classified and normalized"},
        400: {"description": "File is empty or cannot be read"},
        413: {"description": "File exceeds 50 MB limit"},
    },
)
async def discover_upload(
    file: UploadFile = File(description="File to classify: STIX bundle, JSON logs, or threat report"),
) -> DiscoveryRequestOut:
    MAX_BYTES = 50 * 1024 * 1024  # 50 MB

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content):,} bytes). Maximum is 50 MB.",
        )

    req = normalize_file(
        content=content,
        filename=file.filename or "unknown",
        source=InputSource.FILE_UPLOAD,
    )
    result = validate(req)

    return DiscoveryRequestOut.from_request(req, warnings=result.warnings)


@router.get(
    "/discover/types",
    summary="List all supported input types",
    description="Returns all input types the engine can detect and process.",
)
def list_input_types() -> dict:
    return {
        "total": len(InputType),
        "types": [
            {
                "id":          t.value,
                "category":    _category(t),
                "description": _description(t),
                "examples":    _examples(t),
            }
            for t in InputType
        ],
    }


# ── Helpers ────────────────────────────────────────────────────────────

def _category(t: InputType) -> str:
    ioc_types = {InputType.IP_ADDRESS, InputType.DOMAIN, InputType.URL, InputType.FILE_HASH}
    file_types = {InputType.JSON_FILE, InputType.STIX_BUNDLE, InputType.CYBER_REPORT}
    structured = {InputType.CVE_ID, InputType.MALWARE_NAME,
                  InputType.APT_GROUP, InputType.MITRE_TECHNIQUE}
    if t in ioc_types:
        return "IOC"
    if t in file_types:
        return "File Upload"
    if t in structured:
        return "Structured Identifier"
    return "Free-form"


def _description(t: InputType) -> str:
    return {
        InputType.IP_ADDRESS:       "IPv4 or IPv6 address (e.g., 203.0.113.42)",
        InputType.DOMAIN:           "Fully-qualified domain name (e.g., malware-c2.ru)",
        InputType.URL:              "Full URL with protocol (e.g., https://evil.com/payload.exe)",
        InputType.FILE_HASH:        "MD5, SHA1, SHA256, or SHA512 file hash",
        InputType.CVE_ID:           "NIST CVE identifier (e.g., CVE-2021-44228)",
        InputType.MALWARE_NAME:     "Malware family name (e.g., WannaCry, Emotet)",
        InputType.APT_GROUP:        "APT group designation (e.g., APT28, Lazarus Group)",
        InputType.MITRE_TECHNIQUE:  "MITRE ATT&CK ID: technique (T####), tactic (TA####), group (G####)",
        InputType.NATURAL_LANGUAGE: "Free-form incident description or analyst query",
        InputType.JSON_FILE:        "Uploaded JSON log file or event export",
        InputType.STIX_BUNDLE:      "Uploaded STIX 2.0 or 2.1 threat intelligence bundle",
        InputType.CYBER_REPORT:     "Uploaded plain-text or Markdown threat report",
    }.get(t, "")


def _examples(t: InputType) -> list[str]:
    return {
        InputType.IP_ADDRESS:       ["203.0.113.42", "2001:db8::1"],
        InputType.DOMAIN:           ["malware-c2.ru", "phishing-kit.xyz"],
        InputType.URL:              ["https://evil.com/payload.exe", "ftp://attacker.net/tools"],
        InputType.FILE_HASH:        ["84c82835a5d21bbcf75a61706d8ab549 (MD5)",
                                     "a94a8fe5ccb19ba61c4c0873d391e987fd592b52 (SHA1)"],
        InputType.CVE_ID:           ["CVE-2021-44228", "CVE-2017-0144"],
        InputType.MALWARE_NAME:     ["WannaCry", "Emotet", "Cobalt Strike"],
        InputType.APT_GROUP:        ["APT28", "Lazarus Group", "FIN7"],
        InputType.MITRE_TECHNIQUE:  ["T1059", "T1059.001", "TA0001", "G0007"],
        InputType.NATURAL_LANGUAGE: ["Suspicious PowerShell followed by outbound TOR traffic"],
        InputType.JSON_FILE:        ["events.json", "splunk_export.json"],
        InputType.STIX_BUNDLE:      ["threat-intel.json (STIX 2.1 bundle)"],
        InputType.CYBER_REPORT:     ["apt28-report.txt", "incident-analysis.md"],
    }.get(t, [])
