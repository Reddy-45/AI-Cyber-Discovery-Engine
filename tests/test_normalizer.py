"""
tests/test_normalizer.py — Unit tests for the Input Processing Layer.

Tests cover:
    - All 12 input type detections (positive cases)
    - Validation logic (negative cases, edge cases)
    - Metadata extraction per type
    - File classification (STIX / JSON / report)
    - API endpoint integration (POST /discover, POST /discover/upload)
    - Validation warnings (private IPs, suspicious TLDs, short NL)

These tests exercise the normalizer in complete isolation —
no DB, no engine pipeline, no embeddings.
"""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from discovery.models import InputType, InputSource
from discovery.normalizer import normalize_text, normalize_file, validate
from api.main import app

client = TestClient(app, raise_server_exceptions=True)


# ══════════════════════════════════════════════════════════════════════
# Section A — normalize_text() unit tests
# ══════════════════════════════════════════════════════════════════════

class TestIPAddressDetection:
    def test_ipv4_public(self):
        req = normalize_text("8.8.8.8")   # Google DNS — globally routable
        assert req.input_type == InputType.IP_ADDRESS
        assert req.is_valid
        assert req.metadata["version"] == "ipv4"
        assert req.metadata["is_private"] is False

    def test_ipv4_private(self):
        req = normalize_text("192.168.1.100")
        assert req.input_type == InputType.IP_ADDRESS
        assert req.metadata["is_private"] is True

    def test_ipv4_loopback(self):
        req = normalize_text("127.0.0.1")
        assert req.input_type == InputType.IP_ADDRESS
        assert req.metadata["is_loopback"] is True

    def test_ipv6_full(self):
        req = normalize_text("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert req.input_type == InputType.IP_ADDRESS
        assert req.metadata["version"] == "ipv6"

    def test_ipv6_compressed(self):
        req = normalize_text("::1")
        assert req.input_type == InputType.IP_ADDRESS
        assert req.metadata["is_loopback"] is True

    def test_normalized_value_is_canonical(self):
        req = normalize_text("192.168.1.1")   # plain valid IPv4 without leading zeros
        assert req.input_type == InputType.IP_ADDRESS

    def test_invalid_ip_does_not_match(self):
        req = normalize_text("999.999.999.999")
        assert req.input_type != InputType.IP_ADDRESS

    def test_routing_hint(self):
        req = normalize_text("10.0.0.1")
        assert req.routing_hint == "ioc_ip"


class TestDomainDetection:
    def test_simple_domain(self):
        req = normalize_text("malware-c2.ru")
        assert req.input_type == InputType.DOMAIN
        assert req.is_valid
        assert req.metadata["tld"] == ".ru"
        assert req.metadata["apex_domain"] == "malware-c2.ru"

    def test_subdomain(self):
        req = normalize_text("api.example.com")
        assert req.input_type == InputType.DOMAIN
        assert req.metadata["subdomain"] == "api"
        assert req.metadata["apex_domain"] == "example.com"

    def test_domain_normalized_lowercase(self):
        req = normalize_text("EVIL.COM")
        assert req.input_type == InputType.DOMAIN
        assert req.normalized_value == "evil.com"

    def test_url_takes_priority_over_domain(self):
        req = normalize_text("https://evil.com")
        assert req.input_type == InputType.URL  # URL detector runs first

    def test_routing_hint(self):
        req = normalize_text("phishing.xyz")
        assert req.routing_hint == "ioc_domain"


class TestURLDetection:
    def test_https_url(self):
        req = normalize_text("https://evil.com/payload.exe")
        assert req.input_type == InputType.URL
        assert req.metadata["scheme"] == "https"
        assert req.metadata["host"] == "evil.com"
        assert req.metadata["path"] == "/payload.exe"

    def test_http_url(self):
        req = normalize_text("http://c2.attacker.net:8080/beacon")
        assert req.input_type == InputType.URL
        assert req.metadata["port"] == 8080

    def test_ftp_url(self):
        req = normalize_text("ftp://attacker.net/tools/kit.tar.gz")
        assert req.input_type == InputType.URL
        assert req.metadata["scheme"] == "ftp"

    def test_url_normalized_lowercase(self):
        req = normalize_text("HTTPS://Evil.COM/Path")
        assert req.input_type == InputType.URL
        assert req.normalized_value == "https://evil.com/path"

    def test_routing_hint(self):
        req = normalize_text("https://example.com")
        assert req.routing_hint == "ioc_url"


class TestFileHashDetection:
    MD5    = "84c82835a5d21bbcf75a61706d8ab549"
    SHA1   = "a94a8fe5ccb19ba61c4c0873d391e987fd592b52"
    SHA256 = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
    SHA512 = ("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
               "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e")

    def test_md5(self):
        req = normalize_text(self.MD5)
        assert req.input_type == InputType.FILE_HASH
        assert req.metadata["algorithm"] == "md5"
        assert req.metadata["length"] == 32

    def test_sha1(self):
        req = normalize_text(self.SHA1)
        assert req.input_type == InputType.FILE_HASH
        assert req.metadata["algorithm"] == "sha1"

    def test_sha256(self):
        req = normalize_text(self.SHA256)
        assert req.input_type == InputType.FILE_HASH
        assert req.metadata["algorithm"] == "sha256"

    def test_sha512(self):
        req = normalize_text(self.SHA512)
        assert req.input_type == InputType.FILE_HASH
        assert req.metadata["algorithm"] == "sha512"

    def test_hash_normalized_lowercase(self):
        req = normalize_text(self.MD5.upper())
        assert req.input_type == InputType.FILE_HASH
        assert req.normalized_value == self.MD5.lower()

    def test_routing_hint(self):
        req = normalize_text(self.SHA256)
        assert req.routing_hint == "ioc_hash"

    def test_non_hex_not_hash(self):
        req = normalize_text("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")  # 32 chars, not hex
        assert req.input_type != InputType.FILE_HASH


class TestCVEDetection:
    def test_standard_cve(self):
        req = normalize_text("CVE-2021-44228")
        assert req.input_type == InputType.CVE_ID
        assert req.metadata["year"] == 2021
        assert req.metadata["sequence"] == "44228"
        assert req.normalized_value == "CVE-2021-44228"

    def test_lowercase_cve(self):
        req = normalize_text("cve-2017-0144")
        assert req.input_type == InputType.CVE_ID
        assert req.normalized_value == "CVE-2017-0144"

    def test_short_sequence(self):
        req = normalize_text("CVE-2020-0601")
        assert req.input_type == InputType.CVE_ID
        assert req.metadata["year"] == 2020

    def test_routing_hint(self):
        req = normalize_text("CVE-2022-30190")
        assert req.routing_hint == "cve_lookup"

    def test_invalid_cve_too_short(self):
        req = normalize_text("CVE-2021-123")
        assert req.input_type != InputType.CVE_ID


class TestMITREDetection:
    def test_technique(self):
        req = normalize_text("T1059")
        assert req.input_type == InputType.MITRE_TECHNIQUE
        assert req.metadata["id_type"] == "technique"
        assert req.metadata["has_subtechnique"] is False

    def test_subtechnique(self):
        req = normalize_text("T1059.001")
        assert req.input_type == InputType.MITRE_TECHNIQUE
        assert req.metadata["has_subtechnique"] is True
        assert req.metadata["parent_id"] == "T1059"

    def test_tactic(self):
        req = normalize_text("TA0001")
        assert req.input_type == InputType.MITRE_TECHNIQUE
        assert req.metadata["id_type"] == "tactic"

    def test_group(self):
        req = normalize_text("G0007")
        assert req.input_type == InputType.MITRE_TECHNIQUE
        assert req.metadata["id_type"] == "group"

    def test_software(self):
        req = normalize_text("S0002")
        assert req.input_type == InputType.MITRE_TECHNIQUE
        assert req.metadata["id_type"] == "software"

    def test_normalized_uppercase(self):
        req = normalize_text("t1059.001")
        assert req.normalized_value == "T1059.001"

    def test_routing_technique(self):
        req = normalize_text("T1566")
        assert req.routing_hint == "mitre_technique"

    def test_routing_tactic(self):
        req = normalize_text("TA0002")
        assert req.routing_hint == "mitre_tactic"


class TestAPTGroupDetection:
    def test_numbered_apt(self):
        req = normalize_text("APT28")
        assert req.input_type == InputType.APT_GROUP
        assert req.metadata["pattern_type"] == "numbered"
        assert req.normalized_value == "APT28"

    def test_numbered_fin(self):
        req = normalize_text("FIN7")
        assert req.input_type == InputType.APT_GROUP
        assert req.normalized_value == "FIN7"

    def test_known_alias(self):
        req = normalize_text("Lazarus Group")
        assert req.input_type == InputType.APT_GROUP
        assert req.metadata["pattern_type"] == "named"
        assert req.metadata["is_known_alias"] is True

    def test_known_alias_case_insensitive(self):
        req = normalize_text("fancy bear")
        assert req.input_type == InputType.APT_GROUP

    def test_wizard_spider(self):
        req = normalize_text("Wizard Spider")
        assert req.input_type == InputType.APT_GROUP

    def test_routing_hint(self):
        req = normalize_text("APT41")
        assert req.routing_hint == "apt_lookup"


class TestMalwareDetection:
    def test_wannacry(self):
        req = normalize_text("WannaCry")
        assert req.input_type == InputType.MALWARE_NAME
        assert req.metadata["matched_canonical"] == "wannacry"

    def test_emotet(self):
        req = normalize_text("Emotet")
        assert req.input_type == InputType.MALWARE_NAME

    def test_cobalt_strike(self):
        req = normalize_text("Cobalt Strike")
        assert req.input_type == InputType.MALWARE_NAME

    def test_case_insensitive(self):
        req = normalize_text("WANNACRY")
        assert req.input_type == InputType.MALWARE_NAME

    def test_normalized_title_case(self):
        req = normalize_text("mimikatz")
        assert req.input_type == InputType.MALWARE_NAME
        assert req.normalized_value == "Mimikatz"

    def test_routing_hint(self):
        req = normalize_text("Ryuk")
        assert req.routing_hint == "malware_lookup"


class TestNaturalLanguageDetection:
    def test_incident_description(self):
        req = normalize_text(
            "Suspicious PowerShell execution followed by outbound traffic to TOR exit nodes"
        )
        assert req.input_type == InputType.NATURAL_LANGUAGE
        assert req.metadata["word_count"] > 5

    def test_short_unknown_string_falls_through(self):
        req = normalize_text("unknown_stuff_xyz")
        assert req.input_type == InputType.NATURAL_LANGUAGE

    def test_metadata_contains_word_count(self):
        req = normalize_text("check this domain and IP for threats")
        assert "word_count" in req.metadata
        assert "char_count" in req.metadata

    def test_routing_hint(self):
        req = normalize_text("analyze this incident for me")
        assert req.routing_hint == "nlp_query"

    def test_empty_input_is_invalid(self):
        req = normalize_text("   ")
        assert req.is_valid is False

    def test_empty_input_error_message(self):
        req = normalize_text("")
        assert len(req.validation_errors) > 0


# ══════════════════════════════════════════════════════════════════════
# Section B — normalize_file() unit tests
# ══════════════════════════════════════════════════════════════════════

class TestSTIXBundleDetection:
    def _make_bundle(self, version="2.1", objects=None) -> bytes:
        bundle = {
            "type": "bundle",
            "id": "bundle--abc123",
            "spec_version": version,
            "objects": objects or [
                {"type": "indicator", "id": "indicator--xyz", "name": "Test IOC"},
                {"type": "malware", "id": "malware--abc", "name": "TestMalware"},
                {"type": "relationship", "id": "rel--1"},
            ],
        }
        return json.dumps(bundle).encode("utf-8")

    def test_stix_bundle_detected(self):
        req = normalize_file(self._make_bundle(), "threat.json")
        assert req.input_type == InputType.STIX_BUNDLE

    def test_stix_metadata_spec_version(self):
        req = normalize_file(self._make_bundle("2.1"), "bundle.json")
        assert req.metadata["spec_version"] == "2.1"

    def test_stix_metadata_object_count(self):
        req = normalize_file(self._make_bundle(), "bundle.json")
        assert req.metadata["object_count"] == 3

    def test_stix_has_indicators(self):
        req = normalize_file(self._make_bundle(), "bundle.json")
        assert req.metadata["has_indicators"] is True

    def test_stix_has_malware(self):
        req = normalize_file(self._make_bundle(), "bundle.json")
        assert req.metadata["has_malware"] is True

    def test_routing_hint(self):
        req = normalize_file(self._make_bundle(), "bundle.json")
        assert req.routing_hint == "stix_ingest"

    def test_filename_preserved(self):
        req = normalize_file(self._make_bundle(), "apt28-intel.json")
        assert req.filename == "apt28-intel.json"


class TestJSONLogDetection:
    def _make_log(self, record_count=5) -> bytes:
        logs = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "source_ip": f"10.0.0.{i}",
                "event_type": "auth",
                "severity": "high",
                "message": "Failed login",
            }
            for i in range(record_count)
        ]
        return json.dumps(logs).encode("utf-8")

    def test_log_file_detected(self):
        req = normalize_file(self._make_log(), "events.json")
        assert req.input_type == InputType.JSON_FILE

    def test_log_file_detected_as_log(self):
        req = normalize_file(self._make_log(), "events.json")
        assert req.metadata["detected_as"] == "log_file"

    def test_record_count(self):
        req = normalize_file(self._make_log(10), "events.json")
        assert req.metadata["record_count"] == 10

    def test_routing_hint_log_file(self):
        req = normalize_file(self._make_log(), "events.json")
        assert req.routing_hint == "json_log_ingest"


class TestCyberReportDetection:
    def test_txt_report(self):
        content = b"APT28 has been observed using spear-phishing campaigns targeting government entities.\nMimikatz was used for credential harvesting."
        req = normalize_file(content, "report.txt")
        assert req.input_type == InputType.CYBER_REPORT
        assert req.metadata["file_type"] == "txt"

    def test_markdown_report(self):
        content = b"# Threat Report\n## Summary\nThis report covers the Lazarus Group campaign."
        req = normalize_file(content, "report.md")
        assert req.input_type == InputType.CYBER_REPORT

    def test_metadata_char_count(self):
        content = b"A" * 1000
        req = normalize_file(content, "report.txt")
        assert req.metadata["char_count"] == 1000

    def test_empty_file_invalid(self):
        req = normalize_file(b"", "empty.txt")
        assert req.is_valid is False

    def test_routing_hint(self):
        req = normalize_file(b"Some threat report content here.", "report.txt")
        assert req.routing_hint == "report_ingest"


# ══════════════════════════════════════════════════════════════════════
# Section C — validate() unit tests
# ══════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_private_ip_produces_warning(self):
        req = normalize_text("192.168.1.1")
        result = validate(req)
        assert result.is_valid
        assert any("private" in w.lower() for w in result.warnings)

    def test_loopback_ip_produces_warning(self):
        req = normalize_text("127.0.0.1")
        result = validate(req)
        assert any("loopback" in w.lower() for w in result.warnings)

    def test_public_ip_no_warnings(self):
        req = normalize_text("8.8.8.8")   # Google DNS — globally routable, no warnings
        result = validate(req)
        assert result.is_valid
        assert len(result.warnings) == 0

    def test_suspicious_tld_warning(self):
        req = normalize_text("phishing.tk")
        result = validate(req)
        assert any("TLD" in w for w in result.warnings)

    def test_safe_tld_no_warning(self):
        req = normalize_text("microsoft.com")
        result = validate(req)
        assert len(result.warnings) == 0

    def test_short_nl_warning(self):
        req = normalize_text("check ip")  # only 2 words
        result = validate(req)
        assert any("short" in w.lower() for w in result.warnings)

    def test_long_nl_no_warning(self):
        req = normalize_text(
            "The attacker used PowerShell to download a Cobalt Strike beacon "
            "and then performed lateral movement via SMB."
        )
        result = validate(req)
        assert len(result.warnings) == 0


# ══════════════════════════════════════════════════════════════════════
# Section D — API endpoint integration tests
# ══════════════════════════════════════════════════════════════════════

class TestDiscoverEndpoint:
    def test_ip_via_api(self):
        resp = client.post("/api/v1/discover", json={"query": "203.0.113.42"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_type"] == "ip_address"
        assert data["is_valid"] is True

    def test_cve_via_api(self):
        resp = client.post("/api/v1/discover", json={"query": "CVE-2021-44228"})
        assert resp.status_code == 200
        assert resp.json()["input_type"] == "cve_id"

    def test_mitre_via_api(self):
        resp = client.post("/api/v1/discover", json={"query": "T1059.001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_type"] == "mitre_technique"
        assert data["metadata"]["has_subtechnique"] is True

    def test_apt_via_api(self):
        resp = client.post("/api/v1/discover", json={"query": "APT28"})
        assert resp.status_code == 200
        assert resp.json()["input_type"] == "apt_group"

    def test_malware_via_api(self):
        resp = client.post("/api/v1/discover", json={"query": "Emotet"})
        assert resp.status_code == 200
        assert resp.json()["input_type"] == "malware_name"

    def test_nl_via_api(self):
        resp = client.post(
            "/api/v1/discover",
            json={"query": "Attacker exfiltrated data via encrypted DNS tunneling"},
        )
        assert resp.status_code == 200
        assert resp.json()["input_type"] == "natural_language"

    def test_empty_query_returns_400(self):
        resp = client.post("/api/v1/discover", json={"query": "  "})
        assert resp.status_code == 400

    def test_response_has_request_id(self):
        resp = client.post("/api/v1/discover", json={"query": "T1566"})
        assert "request_id" in resp.json()

    def test_response_has_routing_hint(self):
        resp = client.post("/api/v1/discover", json={"query": "FIN7"})
        assert resp.json()["routing_hint"] == "apt_lookup"

    def test_private_ip_returns_warning(self):
        resp = client.post("/api/v1/discover", json={"query": "10.0.0.1"})
        data = resp.json()
        assert data["is_valid"] is True
        assert len(data["warnings"]) > 0

    def test_response_has_metadata(self):
        resp = client.post("/api/v1/discover", json={"query": "CVE-2017-0144"})
        data = resp.json()
        assert "year" in data["metadata"]
        assert data["metadata"]["year"] == 2017


class TestDiscoverUploadEndpoint:
    def test_upload_stix_bundle(self):
        bundle = json.dumps({
            "type": "bundle",
            "id": "bundle--test",
            "spec_version": "2.1",
            "objects": [
                {"type": "indicator", "id": "indicator--1", "name": "test"},
            ],
        }).encode()
        resp = client.post(
            "/api/v1/discover/upload",
            files={"file": ("threat.json", bundle, "application/json")},
        )
        assert resp.status_code == 200
        assert resp.json()["input_type"] == "stix_bundle"

    def test_upload_json_logs(self):
        logs = json.dumps([
            {"timestamp": "2024-01-01T00:00:00Z", "source_ip": "10.0.0.1",
             "event_type": "auth", "severity": "high", "message": "fail"},
        ]).encode()
        resp = client.post(
            "/api/v1/discover/upload",
            files={"file": ("events.json", logs, "application/json")},
        )
        assert resp.status_code == 200
        assert resp.json()["input_type"] == "json_file"

    def test_upload_text_report(self):
        report = b"APT28 used spear-phishing targeting finance sector in Q3 2024."
        resp = client.post(
            "/api/v1/discover/upload",
            files={"file": ("report.txt", report, "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["input_type"] == "cyber_report"

    def test_upload_empty_file_returns_400(self):
        resp = client.post(
            "/api/v1/discover/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_response_has_filename(self):
        report = b"Some threat intelligence content."
        resp = client.post(
            "/api/v1/discover/upload",
            files={"file": ("intel.txt", report, "text/plain")},
        )
        assert resp.json()["filename"] == "intel.txt"


class TestInputTypesEndpoint:
    def test_returns_200(self):
        resp = client.get("/api/v1/discover/types")
        assert resp.status_code == 200

    def test_returns_all_12_types(self):
        data = client.get("/api/v1/discover/types").json()
        assert data["total"] == 12

    def test_each_type_has_required_fields(self):
        data = client.get("/api/v1/discover/types").json()
        for t in data["types"]:
            assert "id" in t
            assert "category" in t
            assert "description" in t
            assert "examples" in t
