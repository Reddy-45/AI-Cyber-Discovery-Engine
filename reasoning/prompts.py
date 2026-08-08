"""
reasoning/prompts.py — Reusable Prompt Templates for the AI Reasoning Layer.

Builds a structured, context-rich prompt for Ollama (or any local LLM)
from a fully-populated InvestigationContext.

Five investigation types, each with a tailored system instruction and
a data-injection section:
    1. IOC Investigation     (IP, domain, URL, file hash)
    2. CVE Investigation     (CVE identifiers)
    3. Malware Investigation (malware family names)
    4. APT Investigation     (threat actor / APT group names)
    5. Incident Investigation (natural-language / STIX / report uploads)

Output format:
    Prompts ask the LLM for a JSON object with keys that map directly to
    InvestigationReport sub-models.  The fallback template reasoner uses
    the same PromptBuilder to extract context — it just doesn't call an LLM.

Usage:
    prompt = PromptBuilder.build(context)
    # Pass prompt to OllamaClient.generate(prompt, model)
"""

from __future__ import annotations

from discovery.aggregator import InvestigationContext
from discovery.models import InputType

# ── Constants ─────────────────────────────────────────────────────────
_MAX_EVIDENCE_LINES = 10   # cap evidence list to keep prompts <2 KB


# ══════════════════════════════════════════════════════════════════════
# System instruction (shared across all investigation types)
# ══════════════════════════════════════════════════════════════════════

_SYSTEM = """\
You are a senior SOC (Security Operations Centre) analyst with 15 years of \
experience in threat intelligence and incident response. \
You receive structured threat intelligence data and produce a concise, \
accurate, professional investigation report in JSON.

RULES
- Respond ONLY with valid JSON — no markdown fences, no explanations outside the JSON.
- Do not invent threat actor names, CVE IDs, or malware names not present in the data.
- If data is insufficient to assess a field, use "UNKNOWN" or an empty list [].
- Be precise and professional. Avoid generic filler sentences.

OUTPUT SCHEMA (produce exactly these keys):
{
  "headline": "<one sentence, e.g. WannaCry ransomware detected — HIGH risk>",
  "threat_level": "<INFORMATIONAL | LOW | MEDIUM | HIGH | CRITICAL>",
  "summary": "<2-4 sentence executive summary for a non-technical audience>",
  "key_findings": ["<finding 1>", "<finding 2>", "..."],
  "risk_score": <float 0.0-10.0>,
  "threat_actors": ["<name>"],
  "attack_techniques": ["<T-ID: name>"],
  "exploited_cves": ["<CVE-ID>"],
  "kill_chain": [
    {"phase": "<phase>", "tactics": ["<tactic>"], "techniques": ["<T-ID>"], "evidence": ["<string>"]}
  ],
  "recommendations": [
    {"priority": "<IMMEDIATE|SHORT_TERM|LONG_TERM>", "category": "<DETECTION|MITIGATION|RESPONSE|HARDENING|MONITORING>", "action": "<imperative>", "rationale": "<why>"}
  ]
}
"""


# ══════════════════════════════════════════════════════════════════════
# Type-specific preamble templates
# ══════════════════════════════════════════════════════════════════════

def _ioc_preamble(ctx: InvestigationContext) -> str:
    return (
        "INVESTIGATION TYPE: IOC Analysis\n"
        f"QUERY: {ctx.query_summary or 'IOC investigation'}\n"
        "The analyst has submitted a malicious indicator (IP / domain / URL / hash). "
        "Determine the threat posed, attribution, and recommended defensive actions."
    )


def _cve_preamble(ctx: InvestigationContext) -> str:
    cve_list = ", ".join(c.cve_id for c in ctx.matched_cves) or "unknown"
    return (
        "INVESTIGATION TYPE: CVE Vulnerability Analysis\n"
        f"QUERY: {ctx.query_summary or cve_list}\n"
        "The analyst is investigating a specific CVE identifier. "
        "Assess the severity, exploitation status, associated threat actors, and patch urgency."
    )


def _malware_preamble(ctx: InvestigationContext) -> str:
    names = ", ".join(m.name for m in ctx.matched_malware) or "unknown"
    return (
        "INVESTIGATION TYPE: Malware Analysis\n"
        f"QUERY: {ctx.query_summary or names}\n"
        "The analyst is investigating a malware family or sample. "
        "Identify TTPs, attributed threat actors, exploited CVEs, and mitigation steps."
    )


def _apt_preamble(ctx: InvestigationContext) -> str:
    names = ", ".join(a.name for a in ctx.matched_apt_groups) or "unknown"
    return (
        "INVESTIGATION TYPE: APT Threat Actor Analysis\n"
        f"QUERY: {ctx.query_summary or names}\n"
        "The analyst is investigating a nation-state or organised threat actor (APT group). "
        "Summarise their capabilities, known campaigns, and defensive recommendations."
    )


def _incident_preamble(ctx: InvestigationContext) -> str:
    return (
        "INVESTIGATION TYPE: Incident Investigation\n"
        f"QUERY: {ctx.query_summary or 'Incident investigation'}\n"
        "The analyst has submitted a natural-language incident description, STIX bundle, "
        "or threat report for investigation. "
        "Synthesise all available intelligence into an actionable incident report."
    )


_PREAMBLE_MAP = {
    InputType.IP_ADDRESS.value:      _ioc_preamble,
    InputType.DOMAIN.value:          _ioc_preamble,
    InputType.URL.value:             _ioc_preamble,
    InputType.FILE_HASH.value:       _ioc_preamble,
    InputType.CVE_ID.value:          _cve_preamble,
    InputType.MALWARE_NAME.value:    _malware_preamble,
    InputType.APT_GROUP.value:       _apt_preamble,
    InputType.MITRE_TECHNIQUE.value: _incident_preamble,
    InputType.NATURAL_LANGUAGE.value: _incident_preamble,
    InputType.JSON_FILE.value:       _incident_preamble,
    InputType.STIX_BUNDLE.value:     _incident_preamble,
    InputType.CYBER_REPORT.value:    _incident_preamble,
}


# ══════════════════════════════════════════════════════════════════════
# Data section builder — injects structured findings into the prompt
# ══════════════════════════════════════════════════════════════════════

def _build_data_section(ctx: InvestigationContext) -> str:
    lines: list[str] = ["", "INTELLIGENCE DATA", "─" * 50]

    # ── MITRE Techniques ─────────────────────────────────────────────
    if ctx.matched_techniques:
        lines.append(f"MITRE ATT&CK Techniques ({len(ctx.matched_techniques)}):")
        for t in ctx.matched_techniques[:8]:
            lines.append(f"  • {t.technique_id}: {t.technique_name} [{t.tactic}]")

    # ── Malware ──────────────────────────────────────────────────────
    if ctx.matched_malware:
        lines.append(f"Malware Families ({len(ctx.matched_malware)}):")
        for m in ctx.matched_malware:
            cves = ", ".join(m.cves_exploited[:3]) or "none"
            lines.append(
                f"  • {m.name} ({m.family_type or 'unknown type'}) | "
                f"attributed: {', '.join(m.attributed_to) or 'unknown'} | "
                f"CVEs: {cves}"
            )

    # ── CVEs ─────────────────────────────────────────────────────────
    if ctx.matched_cves:
        lines.append(f"CVEs ({len(ctx.matched_cves)}):")
        for c in ctx.matched_cves:
            kev = " [CISA KEV]" if c.cisa_kev else ""
            lines.append(
                f"  • {c.cve_id} CVSS:{c.cvss_score} {c.severity.upper()}{kev} — {c.name or c.description[:60]}"
            )

    # ── APT Groups ───────────────────────────────────────────────────
    if ctx.matched_apt_groups:
        lines.append(f"APT Groups ({len(ctx.matched_apt_groups)}):")
        for a in ctx.matched_apt_groups:
            lines.append(
                f"  • {a.name} ({a.country or 'origin unknown'}) | "
                f"malware: {', '.join(a.malware_used[:3]) or 'unknown'}"
            )

    # ── IOC Intel ────────────────────────────────────────────────────
    if ctx.matched_iocs:
        lines.append(f"Malicious Indicators ({len(ctx.matched_iocs)}):")
        for ioc in ctx.matched_iocs[:5]:
            tags = ", ".join(ioc.tags) if ioc.tags else "none"
            lines.append(
                f"  • [{ioc.ioc_type.upper()}] {ioc.value} | "
                f"tags: {tags} | confidence: {ioc.confidence:.0%}"
            )

    # ── Evidence strings ─────────────────────────────────────────────
    if ctx.evidence:
        lines.append(f"Supporting Evidence ({min(len(ctx.evidence), _MAX_EVIDENCE_LINES)} of {len(ctx.evidence)}):")
        for ev in ctx.evidence[:_MAX_EVIDENCE_LINES]:
            lines.append(f"  – {ev}")

    # ── Aggregated confidence ─────────────────────────────────────────
    lines.append(f"Aggregated Confidence Score: {ctx.confidence_score:.0%}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

class PromptBuilder:
    """
    Builds Ollama-ready prompts from an InvestigationContext.

    Usage:
        system, user = PromptBuilder.build(context)
        # system → system role instruction
        # user   → user message with preamble + structured data
    """

    @staticmethod
    def build(ctx: InvestigationContext) -> tuple[str, str]:
        """
        Return (system_prompt, user_prompt) for the given context.

        The system prompt sets the analyst role and output schema.
        The user prompt provides the investigation type, query, and all
        intelligence data extracted from the context.
        """
        preamble_fn = _PREAMBLE_MAP.get(
            ctx.input_type,
            _incident_preamble,
        )
        preamble = preamble_fn(ctx)
        data_section = _build_data_section(ctx)
        user_prompt = f"{preamble}\n{data_section}\n\nNow produce the JSON report:"
        return _SYSTEM, user_prompt

    @staticmethod
    def build_flat(ctx: InvestigationContext) -> str:
        """
        Return a single combined prompt string (system + user) for LLM backends
        that accept a single string instead of a chat-style payload.
        """
        system, user = PromptBuilder.build(ctx)
        return f"SYSTEM INSTRUCTION:\n{system}\n\nUSER:\n{user}"
