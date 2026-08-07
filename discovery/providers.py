"""
discovery/providers.py — Knowledge Providers for the Discovery Engine.

Each provider class is responsible for a single knowledge domain.
All providers:
    - Load from local JSON files only (no internet, no APIs, no LLM)
    - Accept a DiscoveryRequest
    - Return a typed result or empty result if the input type is not relevant
    - Are stateless and cache their data on first load

Provider classes:
    MITREProvider        — ATT&CK technique lookup and keyword matching
    ThreatIntelProvider  — Known-bad IP / hash / domain reputation
    MalwareProvider      — Malware family profiles, related TTPs and IOCs
    APTGroupProvider     — APT group attribution and associated techniques
    CVEProvider          — CVE details, severity, exploitation status
    STIXProvider         — Parse uploaded STIX bundle objects into entities

Design principles:
    - Providers do NOT cross-reference each other (that is the correlator's job)
    - Each provider returns only what its own data source knows
    - Missing data returns empty containers, never raises exceptions
    - The engine's enrich.py maps IOCs *within events* — providers map
      *explicit analyst queries* about specific entities. No duplication.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from discovery.models import DiscoveryRequest, InputType
# Reuse engine types rather than redefine them
from engine.models import IOCMatch, IOCType, MITRETechnique

logger = logging.getLogger(__name__)

# ── Project root — data files live relative to here ───────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"


# ══════════════════════════════════════════════════════════════════════
# Provider result models
# ══════════════════════════════════════════════════════════════════════

class MalwareMatch(BaseModel):
    """A malware family matched from the knowledge base."""
    name:               str
    aliases:            list[str]           = Field(default_factory=list)
    family_type:        str                 = ""     # ransomware / trojan / c2 / etc.
    attributed_to:      list[str]           = Field(default_factory=list)
    targeted_sectors:   list[str]           = Field(default_factory=list)
    mitre_techniques:   list[str]           = Field(default_factory=list)
    cves_exploited:     list[str]           = Field(default_factory=list)
    ioc_hashes:         list[str]           = Field(default_factory=list)
    ioc_domains:        list[str]           = Field(default_factory=list)
    description:        str                 = ""
    kill_chain:         list[str]           = Field(default_factory=list)
    confidence:         float               = 1.0


class APTMatch(BaseModel):
    """An APT group matched from the knowledge base."""
    name:               str
    aliases:            list[str]           = Field(default_factory=list)
    country:            str                 = ""
    sponsor:            str                 = ""
    mitre_group_id:     str                 = ""
    targeted_sectors:   list[str]           = Field(default_factory=list)
    mitre_techniques:   list[str]           = Field(default_factory=list)
    malware_used:       list[str]           = Field(default_factory=list)
    cves_exploited:     list[str]           = Field(default_factory=list)
    known_campaigns:    list[str]           = Field(default_factory=list)
    description:        str                 = ""
    confidence:         float               = 1.0


class CVEMatch(BaseModel):
    """A CVE entry matched from the knowledge base."""
    cve_id:             str
    name:               str                 = ""
    cvss_score:         float               = 0.0
    severity:           str                 = ""
    published:          str                 = ""
    affected_products:  list[str]           = Field(default_factory=list)
    description:        str                 = ""
    exploitation_status: str               = ""
    cisa_kev:           bool               = False
    exploit_public:     bool               = False
    related_malware:    list[str]           = Field(default_factory=list)
    related_apt_groups: list[str]           = Field(default_factory=list)
    mitre_techniques:   list[str]           = Field(default_factory=list)
    confidence:         float               = 1.0


class IOCIntelMatch(BaseModel):
    """Threat intelligence hit for an IOC (IP, domain, or hash)."""
    value:              str
    ioc_type:           str                 # "ip", "domain", "hash"
    tags:               list[str]           = Field(default_factory=list)
    malware_family:     str | None          = None
    confidence:         float               = 0.5
    source:             str                 = ""
    is_known_bad:       bool               = False


class ProviderResult(BaseModel):
    """
    Typed output from a single provider.

    All fields are optional — a provider returns only what it found.
    Empty lists are the norm for irrelevant input types.
    """
    source:             str                         # Provider class name
    matched_techniques: list[MITRETechnique]        = Field(default_factory=list)
    matched_malware:    list[MalwareMatch]           = Field(default_factory=list)
    matched_apt_groups: list[APTMatch]              = Field(default_factory=list)
    matched_cves:       list[CVEMatch]              = Field(default_factory=list)
    matched_iocs:       list[IOCIntelMatch]         = Field(default_factory=list)
    evidence:           list[str]                   = Field(default_factory=list)
    # Raw STIX objects extracted from uploaded bundles
    stix_objects:       list[dict[str, Any]]        = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Base provider
# ══════════════════════════════════════════════════════════════════════

class BaseProvider(ABC):
    """Abstract base for all knowledge providers."""

    name: str = "base"

    @abstractmethod
    def query(self, request: DiscoveryRequest) -> ProviderResult:
        """
        Query this provider with a normalized DiscoveryRequest.

        Must never raise. Return an empty ProviderResult if the input
        type is not relevant or data is not found.
        """
        ...

    def _empty(self) -> ProviderResult:
        return ProviderResult(source=self.name)

    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load a JSON file, returning {} on any error."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load %s: %s", path, exc)
            return {}


# ══════════════════════════════════════════════════════════════════════
# MITRE ATT&CK Provider
# ══════════════════════════════════════════════════════════════════════

class MITREProvider(BaseProvider):
    """
    Looks up ATT&CK techniques from the local mitre_attack.json.

    Handles three cases:
    1. Direct ID lookup  (T1059, TA0001)  — exact match
    2. Malware/APT name — returns associated techniques from the technique list
    3. Natural language  — keyword match against technique keyword lists
    """

    name = "MITREProvider"
    _db: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self.__class__._db is None:
            data = self._load_json(_DATA / "mitre_attack.json")
            self.__class__._db = data.get("techniques", [])
        return self.__class__._db  # type: ignore[return-value]

    def query(self, request: DiscoveryRequest) -> ProviderResult:
        db = self._load()
        matched: list[MITRETechnique] = []
        evidence: list[str] = []

        if request.input_type == InputType.MITRE_TECHNIQUE:
            # Direct ID lookup
            qid = request.normalized_value.upper()
            for t in db:
                if t["technique_id"].upper() == qid:
                    matched.append(_to_mitre_technique(t))
                    evidence.append(f"Direct MITRE ID match: {qid}")
                    break

        elif request.input_type in (InputType.NATURAL_LANGUAGE, InputType.CYBER_REPORT):
            # Keyword scan across the whole text
            text = request.normalized_value.lower()
            for t in db:
                if any(kw.lower() in text for kw in t.get("keywords", [])):
                    matched.append(_to_mitre_technique(t))
                    evidence.append(
                        f"Keyword match '{t['technique_id']}' in text"
                    )

        elif request.input_type == InputType.MALWARE_NAME:
            # Techniques are linked via MalwareProvider; MITRE also checks keywords
            text = request.normalized_value.lower()
            for t in db:
                if any(kw.lower() in text for kw in t.get("keywords", [])):
                    matched.append(_to_mitre_technique(t))
                    evidence.append(f"MITRE keyword match for malware '{request.raw_value}'")

        return ProviderResult(
            source=self.name,
            matched_techniques=_dedupe_techniques(matched),
            evidence=evidence,
        )


# ══════════════════════════════════════════════════════════════════════
# Threat Intelligence Provider
# ══════════════════════════════════════════════════════════════════════

class ThreatIntelProvider(BaseProvider):
    """
    Checks known-bad IP addresses, file hashes, and domains from
    local threat_intel.json.

    This operates at the IOC level — it does NOT duplicate engine/enrich.py's
    per-event enrichment. Here we enrich an *explicit analyst query* about a
    specific IOC, not an event stream.
    """

    name = "ThreatIntelProvider"
    _db: dict | None = None

    def _load(self) -> dict:
        if self.__class__._db is None:
            self.__class__._db = self._load_json(_DATA / "threat_intel.json")
        return self.__class__._db  # type: ignore[return-value]

    def query(self, request: DiscoveryRequest) -> ProviderResult:
        db = self._load()
        ioc_hits: list[IOCIntelMatch] = []
        evidence: list[str] = []

        if request.input_type == InputType.IP_ADDRESS:
            val = request.normalized_value
            for entry in db.get("malicious_ips", []):
                if entry["ip"] == val:
                    ioc_hits.append(IOCIntelMatch(
                        value=val,
                        ioc_type="ip",
                        tags=entry.get("tags", []),
                        confidence=entry.get("confidence", 0.5),
                        source=entry.get("source", "local_threat_intel"),
                        is_known_bad=True,
                    ))
                    evidence.append(
                        f"IP {val} found in local threat intel "
                        f"(tags: {entry.get('tags', [])})"
                    )

        elif request.input_type == InputType.DOMAIN:
            val = request.normalized_value
            for entry in db.get("malicious_domains", []):
                if entry["domain"].lower() == val.lower():
                    ioc_hits.append(IOCIntelMatch(
                        value=val,
                        ioc_type="domain",
                        tags=entry.get("tags", []),
                        confidence=entry.get("confidence", 0.5),
                        source=entry.get("source", "local_threat_intel"),
                        is_known_bad=True,
                    ))
                    evidence.append(
                        f"Domain {val} found in local threat intel "
                        f"(tags: {entry.get('tags', [])})"
                    )

        elif request.input_type == InputType.URL:
            # Extract host from URL metadata and check against domain list
            host = request.metadata.get("host", "")
            if host:
                for entry in db.get("malicious_domains", []):
                    if entry["domain"].lower() == host.lower():
                        ioc_hits.append(IOCIntelMatch(
                            value=host,
                            ioc_type="domain",
                            tags=entry.get("tags", []),
                            confidence=entry.get("confidence", 0.5),
                            source=entry.get("source", "local_threat_intel"),
                            is_known_bad=True,
                        ))
                        evidence.append(
                            f"URL host '{host}' matches malicious domain list"
                        )

        elif request.input_type == InputType.FILE_HASH:
            val = request.normalized_value.lower()
            for entry in db.get("malicious_hashes", []):
                if entry["hash"].lower() == val:
                    ioc_hits.append(IOCIntelMatch(
                        value=val,
                        ioc_type="hash",
                        tags=[entry.get("type", "malware")],
                        malware_family=entry.get("malware_family"),
                        confidence=entry.get("confidence", 0.5),
                        source=entry.get("source", "local_threat_intel"),
                        is_known_bad=True,
                    ))
                    evidence.append(
                        f"Hash {val[:12]}... identified as "
                        f"{entry.get('malware_family', 'unknown malware')} "
                        f"(confidence {entry.get('confidence', 0.5):.0%})"
                    )

        return ProviderResult(
            source=self.name,
            matched_iocs=ioc_hits,
            evidence=evidence,
        )


# ══════════════════════════════════════════════════════════════════════
# Malware Knowledge Provider
# ══════════════════════════════════════════════════════════════════════

class MalwareProvider(BaseProvider):
    """
    Looks up malware family profiles from local malware_families.json.

    Matches on:
    - Direct name query (analyst typed "WannaCry")
    - File hash (cross-reference IOC hashes in family profiles)
    - Domain (cross-reference IOC domains in family profiles)
    - CVE (malware that exploits a given CVE)
    - Natural language / cyber report (keyword match on names/aliases)
    """

    name = "MalwareProvider"
    _db: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self.__class__._db is None:
            data = self._load_json(_DATA / "malware_families.json")
            self.__class__._db = data.get("families", [])
        return self.__class__._db  # type: ignore[return-value]

    def _to_match(self, family: dict, confidence: float = 1.0) -> MalwareMatch:
        ioc = family.get("ioc_patterns", {})
        return MalwareMatch(
            name=family["name"],
            aliases=family.get("aliases", []),
            family_type=family.get("type", ""),
            attributed_to=family.get("attributed_to", []),
            targeted_sectors=family.get("targeted_sectors", []),
            mitre_techniques=family.get("mitre_techniques", []),
            cves_exploited=family.get("cves_exploited", []),
            ioc_hashes=ioc.get("file_hashes_md5", []),
            ioc_domains=ioc.get("domains", []),
            description=family.get("description", ""),
            kill_chain=family.get("kill_chain", []),
            confidence=confidence,
        )

    def query(self, request: DiscoveryRequest) -> ProviderResult:
        db = self._load()
        matched: list[MalwareMatch] = []
        evidence: list[str] = []

        if request.input_type == InputType.MALWARE_NAME:
            # Direct name match (normalized value is title-cased by normalizer)
            query_lower = request.normalized_value.lower()
            for family in db:
                all_names = [family["name"].lower()] + [
                    a.lower() for a in family.get("aliases", [])
                ]
                if query_lower in all_names:
                    matched.append(self._to_match(family))
                    evidence.append(
                        f"Exact match: '{request.raw_value}' → {family['name']}"
                    )

        elif request.input_type == InputType.FILE_HASH:
            val = request.normalized_value.lower()
            for family in db:
                known_hashes = [
                    h.lower()
                    for h in family.get("ioc_patterns", {}).get("file_hashes_md5", [])
                ]
                if val in known_hashes:
                    matched.append(self._to_match(family, confidence=0.98))
                    evidence.append(
                        f"Hash {val[:12]}... matches IOC pattern for {family['name']}"
                    )

        elif request.input_type == InputType.DOMAIN:
            val = request.normalized_value.lower()
            for family in db:
                known_domains = [
                    d.lower()
                    for d in family.get("ioc_patterns", {}).get("domains", [])
                ]
                if val in known_domains:
                    matched.append(self._to_match(family, confidence=0.90))
                    evidence.append(
                        f"Domain '{val}' matches IOC pattern for {family['name']}"
                    )

        elif request.input_type == InputType.CVE_ID:
            cve = request.normalized_value.upper()
            for family in db:
                if cve in [c.upper() for c in family.get("cves_exploited", [])]:
                    matched.append(self._to_match(family, confidence=0.85))
                    evidence.append(
                        f"{family['name']} exploits {cve}"
                    )

        elif request.input_type in (InputType.NATURAL_LANGUAGE, InputType.CYBER_REPORT):
            text = request.normalized_value.lower()
            for family in db:
                all_names = [family["name"].lower()] + [
                    a.lower() for a in family.get("aliases", [])
                ]
                for name in all_names:
                    if name in text:
                        matched.append(self._to_match(family, confidence=0.75))
                        evidence.append(
                            f"Malware '{family['name']}' mentioned in text"
                        )
                        break   # one match per family

        return ProviderResult(
            source=self.name,
            matched_malware=_dedupe_malware(matched),
            evidence=evidence,
        )


# ══════════════════════════════════════════════════════════════════════
# APT Group Provider
# ══════════════════════════════════════════════════════════════════════

class APTGroupProvider(BaseProvider):
    """
    Looks up APT group profiles from local apt_groups.json.

    Matches on:
    - Direct group name / alias query
    - MITRE group ID (G0007)
    - Malware name (groups known to use a particular tool)
    - CVE (groups known to exploit a given CVE)
    - Natural language (group name mentioned in description)
    """

    name = "APTGroupProvider"
    _db: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self.__class__._db is None:
            data = self._load_json(_DATA / "apt_groups.json")
            self.__class__._db = data.get("groups", [])
        return self.__class__._db  # type: ignore[return-value]

    def _to_match(self, group: dict, confidence: float = 1.0) -> APTMatch:
        return APTMatch(
            name=group["name"],
            aliases=group.get("aliases", []),
            country=group.get("country", ""),
            sponsor=group.get("sponsor", ""),
            mitre_group_id=group.get("mitre_group_id", ""),
            targeted_sectors=group.get("targeted_sectors", []),
            mitre_techniques=group.get("mitre_techniques", []),
            malware_used=group.get("malware_used", []),
            cves_exploited=group.get("cves_exploited", []),
            known_campaigns=group.get("known_campaigns", []),
            description=group.get("description", ""),
            confidence=min(confidence, group.get("confidence", 1.0)),
        )

    def query(self, request: DiscoveryRequest) -> ProviderResult:
        db = self._load()
        matched: list[APTMatch] = []
        evidence: list[str] = []

        if request.input_type == InputType.APT_GROUP:
            query_lower = request.normalized_value.lower()
            for group in db:
                all_names = [group["name"].lower()] + [
                    a.lower() for a in group.get("aliases", [])
                ]
                if query_lower in all_names:
                    matched.append(self._to_match(group))
                    evidence.append(
                        f"Exact match: '{request.raw_value}' → {group['name']}"
                    )

        elif request.input_type == InputType.MITRE_TECHNIQUE:
            # MITRE group ID (G####)
            qid = request.normalized_value.upper()
            if request.metadata.get("id_type") == "group":
                for group in db:
                    if group.get("mitre_group_id", "").upper() == qid:
                        matched.append(self._to_match(group))
                        evidence.append(
                            f"MITRE Group ID {qid} → {group['name']}"
                        )

        elif request.input_type == InputType.MALWARE_NAME:
            malware_lower = request.normalized_value.lower()
            for group in db:
                group_malware = [m.lower() for m in group.get("malware_used", [])]
                if malware_lower in group_malware:
                    matched.append(self._to_match(group, confidence=0.80))
                    evidence.append(
                        f"{group['name']} is known to use {request.raw_value}"
                    )

        elif request.input_type == InputType.CVE_ID:
            cve = request.normalized_value.upper()
            for group in db:
                if cve in [c.upper() for c in group.get("cves_exploited", [])]:
                    matched.append(self._to_match(group, confidence=0.85))
                    evidence.append(
                        f"{group['name']} exploits {cve}"
                    )

        elif request.input_type in (InputType.NATURAL_LANGUAGE, InputType.CYBER_REPORT):
            text = request.normalized_value.lower()
            for group in db:
                all_names = [group["name"].lower()] + [
                    a.lower() for a in group.get("aliases", [])
                ]
                for name in all_names:
                    if name in text:
                        matched.append(self._to_match(group, confidence=0.75))
                        evidence.append(
                            f"APT group '{group['name']}' mentioned in text"
                        )
                        break

        return ProviderResult(
            source=self.name,
            matched_apt_groups=_dedupe_apt(matched),
            evidence=evidence,
        )


# ══════════════════════════════════════════════════════════════════════
# CVE Provider
# ══════════════════════════════════════════════════════════════════════

class CVEProvider(BaseProvider):
    """
    Looks up CVE details from local cve_db.json.

    Matches on:
    - Direct CVE ID lookup
    - Natural language / cyber report keyword matching
    - Malware name (CVEs exploited by that malware via cross-reference)
    """

    name = "CVEProvider"
    _db: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self.__class__._db is None:
            data = self._load_json(_DATA / "cve_db.json")
            self.__class__._db = data.get("cves", [])
        return self.__class__._db  # type: ignore[return-value]

    def _to_match(self, cve: dict, confidence: float = 1.0) -> CVEMatch:
        return CVEMatch(
            cve_id=cve["cve_id"],
            name=cve.get("name", ""),
            cvss_score=cve.get("cvss_score", 0.0),
            severity=cve.get("severity", ""),
            published=cve.get("published", ""),
            affected_products=cve.get("affected_products", []),
            description=cve.get("description", ""),
            exploitation_status=cve.get("exploitation_status", ""),
            cisa_kev=cve.get("cisa_kev", False),
            exploit_public=cve.get("exploit_public", False),
            related_malware=cve.get("related_malware", []),
            related_apt_groups=cve.get("related_apt_groups", []),
            mitre_techniques=cve.get("mitre_techniques", []),
            confidence=confidence,
        )

    def query(self, request: DiscoveryRequest) -> ProviderResult:
        db = self._load()
        matched: list[CVEMatch] = []
        evidence: list[str] = []

        if request.input_type == InputType.CVE_ID:
            target = request.normalized_value.upper()
            for cve in db:
                if cve["cve_id"].upper() == target:
                    matched.append(self._to_match(cve))
                    evidence.append(
                        f"CVE {target} found: {cve.get('name', '')} "
                        f"(CVSS {cve.get('cvss_score', '?')})"
                    )
                    break

        elif request.input_type in (InputType.NATURAL_LANGUAGE, InputType.CYBER_REPORT):
            text = request.normalized_value.lower()
            # Check inline CVE-XXXX-XXXXX patterns first
            found_ids = set(re.findall(r'cve-\d{4}-\d{4,7}', text))
            for cve in db:
                if cve["cve_id"].lower() in found_ids:
                    matched.append(self._to_match(cve, confidence=0.95))
                    evidence.append(f"CVE ID {cve['cve_id']} found in text")
                # Keyword match
                elif any(kw.lower() in text for kw in cve.get("keywords", [])):
                    matched.append(self._to_match(cve, confidence=0.70))
                    evidence.append(
                        f"Keywords for {cve['cve_id']} ({cve.get('name','')}) "
                        f"found in text"
                    )

        elif request.input_type == InputType.MALWARE_NAME:
            # CVEs exploited by this malware (cross-reference from CVE side)
            malware_lower = request.normalized_value.lower()
            for cve in db:
                related = [m.lower() for m in cve.get("related_malware", [])]
                if malware_lower in related:
                    matched.append(self._to_match(cve, confidence=0.80))
                    evidence.append(
                        f"{cve['cve_id']} is exploited by {request.raw_value}"
                    )

        return ProviderResult(
            source=self.name,
            matched_cves=_dedupe_cves(matched),
            evidence=evidence,
        )


# ══════════════════════════════════════════════════════════════════════
# STIX Provider
# ══════════════════════════════════════════════════════════════════════

class STIXProvider(BaseProvider):
    """
    Parses an uploaded STIX 2.x bundle and extracts typed entities.

    Handles:
    - indicator objects     → IOC entries
    - malware objects       → MalwareMatch entries
    - threat-actor objects  → APTMatch entries
    - vulnerability objects → CVEMatch entries (if CVE IDs present)
    - relationship objects  → evidence strings

    No external stix2 library needed — we parse the raw JSON structure.
    """

    name = "STIXProvider"

    def query(self, request: DiscoveryRequest) -> ProviderResult:
        if request.input_type != InputType.STIX_BUNDLE:
            return self._empty()

        try:
            bundle = json.loads(request.normalized_value)
        except (json.JSONDecodeError, ValueError):
            return ProviderResult(
                source=self.name,
                evidence=["STIX bundle could not be parsed as JSON"],
            )

        objects: list[dict] = bundle.get("objects", [])
        ioc_hits: list[IOCIntelMatch] = []
        malware_hits: list[MalwareMatch] = []
        apt_hits: list[APTMatch] = []
        cve_hits: list[CVEMatch] = []
        evidence: list[str] = []
        stix_objects: list[dict] = []

        _RE_IP  = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        _RE_HASH = re.compile(r"\b[0-9a-fA-F]{32,64}\b")
        _RE_CVE_ID = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

        for obj in objects:
            obj_type = obj.get("type", "")
            stix_objects.append({"type": obj_type, "id": obj.get("id", ""),
                                  "name": obj.get("name", "")})

            if obj_type == "indicator":
                pattern = obj.get("pattern", "")
                name = obj.get("name", "indicator")
                # Extract IPs from pattern
                for ip in _RE_IP.findall(pattern):
                    ioc_hits.append(IOCIntelMatch(
                        value=ip, ioc_type="ip",
                        tags=["stix_indicator"],
                        confidence=0.85,
                        source=f"STIX bundle: {name}",
                        is_known_bad=True,
                    ))
                # Extract hashes from pattern
                for h in _RE_HASH.findall(pattern):
                    ioc_hits.append(IOCIntelMatch(
                        value=h.lower(), ioc_type="hash",
                        tags=["stix_indicator"],
                        confidence=0.85,
                        source=f"STIX bundle: {name}",
                        is_known_bad=True,
                    ))
                evidence.append(f"STIX indicator: {name}")

            elif obj_type == "malware":
                malware_hits.append(MalwareMatch(
                    name=obj.get("name", "Unknown"),
                    aliases=obj.get("aliases", []),
                    family_type=obj.get("malware_types", ["unknown"])[0]
                    if obj.get("malware_types") else "unknown",
                    description=obj.get("description", ""),
                    confidence=0.80,
                ))
                evidence.append(f"STIX malware: {obj.get('name', 'unknown')}")

            elif obj_type == "threat-actor":
                apt_hits.append(APTMatch(
                    name=obj.get("name", "Unknown"),
                    aliases=obj.get("aliases", []),
                    country=obj.get("country", ""),
                    description=obj.get("description", ""),
                    confidence=0.75,
                ))
                evidence.append(f"STIX threat-actor: {obj.get('name', 'unknown')}")

            elif obj_type == "vulnerability":
                external = obj.get("external_references", [])
                cve_id = next(
                    (r.get("external_id", "")
                     for r in external
                     if r.get("source_name") == "cve"),
                    "",
                )
                if cve_id:
                    cve_hits.append(CVEMatch(
                        cve_id=cve_id.upper(),
                        name=obj.get("name", ""),
                        description=obj.get("description", ""),
                        confidence=0.80,
                    ))
                    evidence.append(f"STIX vulnerability: {cve_id}")

            elif obj_type == "relationship":
                src = obj.get("source_ref", "")
                rel = obj.get("relationship_type", "")
                tgt = obj.get("target_ref", "")
                evidence.append(f"STIX relationship: {src} --[{rel}]--> {tgt}")

        return ProviderResult(
            source=self.name,
            matched_iocs=ioc_hits,
            matched_malware=malware_hits,
            matched_apt_groups=apt_hits,
            matched_cves=cve_hits,
            evidence=evidence,
            stix_objects=stix_objects,
        )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _to_mitre_technique(t: dict) -> MITRETechnique:
    return MITRETechnique(
        technique_id=t["technique_id"],
        technique_name=t["technique_name"],
        tactic=t["tactic"],
        tactic_order=t.get("tactic_order", 0),
    )


def _dedupe_techniques(items: list[MITRETechnique]) -> list[MITRETechnique]:
    seen: set[str] = set()
    out: list[MITRETechnique] = []
    for t in items:
        if t.technique_id not in seen:
            seen.add(t.technique_id)
            out.append(t)
    return out


def _dedupe_malware(items: list[MalwareMatch]) -> list[MalwareMatch]:
    seen: set[str] = set()
    out: list[MalwareMatch] = []
    for m in items:
        key = m.name.lower()
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _dedupe_apt(items: list[APTMatch]) -> list[APTMatch]:
    seen: set[str] = set()
    out: list[APTMatch] = []
    for a in items:
        key = a.name.lower()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def _dedupe_cves(items: list[CVEMatch]) -> list[CVEMatch]:
    seen: set[str] = set()
    out: list[CVEMatch] = []
    for c in items:
        if c.cve_id not in seen:
            seen.add(c.cve_id)
            out.append(c)
    return out
