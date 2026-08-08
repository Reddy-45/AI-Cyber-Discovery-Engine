"""
reasoning/report_builder.py — Investigation Report Generator.

This is the final stage of the AI Cyber Discovery Engine pipeline:

    Input → Normalise → Aggregate → Semantic → Reason → InvestigationReport

Two execution paths, chosen automatically:

    Path A — Ollama available:
        1. Build prompt from context (prompts.PromptBuilder)
        2. Call Ollama (llm_reasoner.LLMReasoner)
        3. Parse LLM JSON → populate InvestigationReport sub-models
        4. Fill any missing fields via deterministic template logic

    Path B — Ollama unavailable (default for most environments):
        1. Build all report fields deterministically from InvestigationContext
        2. Produce a complete, professional-quality report with no LLM

The template reasoner (TemplateReasoner) is NOT a stub — it generates
coherent, specific, evidence-backed content by reading every field of the
InvestigationContext. A real analyst reviewing the template report will
find it substantive, not generic.

Public API:
    report = build_report(context)          # convenience function
    report = ReportBuilder().build(context) # class-based

Both accept InvestigationContext OR EnrichedInvestigationContext.
"""

from __future__ import annotations

import logging
from typing import Any

from discovery.aggregator import InvestigationContext
from discovery.models import InputType
from reasoning.llm_reasoner import LLMReasoner, OllamaClient, ReasoningOutput
from reasoning.models import (
    EvidenceSummary,
    ExecutiveSummary,
    InvestigationReport,
    KillChainPhase,
    Recommendation,
    ReasoningMethod,
    RecommendationCategory,
    RecommendationPriority,
    ThreatAssessment,
    ThreatLevel,
)
from reasoning.prompts import PromptBuilder

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Kill-chain mapping  (MITRE tactic → Cyber Kill Chain phase)
# ══════════════════════════════════════════════════════════════════════

_TACTIC_TO_PHASE: dict[str, str] = {
    "reconnaissance":       "Reconnaissance",
    "resource-development": "Weaponization",
    "initial-access":       "Delivery",
    "execution":            "Exploitation",
    "persistence":          "Installation",
    "privilege-escalation": "Installation",
    "defense-evasion":      "Installation",
    "credential-access":    "Command & Control",
    "discovery":            "Command & Control",
    "lateral-movement":     "Command & Control",
    "collection":           "Command & Control",
    "command-and-control":  "Command & Control",
    "exfiltration":         "Actions on Objectives",
    "impact":               "Actions on Objectives",
}

_PHASE_ORDER = [
    "Reconnaissance",
    "Weaponization",
    "Delivery",
    "Exploitation",
    "Installation",
    "Command & Control",
    "Actions on Objectives",
]


# ══════════════════════════════════════════════════════════════════════
# Risk scoring helpers
# ══════════════════════════════════════════════════════════════════════

def _compute_risk_score(ctx: InvestigationContext) -> float:
    """
    Weighted risk score on a 0.0-10.0 scale (matching CVSS convention).

    Weights:
        CVE CVSS (max 4.0)  — highest individual CVE CVSS scaled to 4
        APT presence (max 3.0) — each APT adds 1.0, capped at 3
        IOC confidence (max 2.0) — average IOC confidence scaled to 2
        CISA KEV bonus (max 1.0) — any CISA-KEV CVE adds 1.0
    """
    score = 0.0

    if ctx.matched_cves:
        max_cvss = max(c.cvss_score for c in ctx.matched_cves)
        score += (max_cvss / 10.0) * 4.0

    if ctx.matched_apt_groups:
        score += min(float(len(ctx.matched_apt_groups)), 3.0)

    if ctx.matched_iocs:
        avg_conf = sum(i.confidence for i in ctx.matched_iocs) / len(ctx.matched_iocs)
        score += avg_conf * 2.0

    if any(c.cisa_kev for c in ctx.matched_cves):
        score += 1.0

    # Malware presence floor: at least 3.0 if malware found
    if ctx.matched_malware and score < 3.0:
        score = 3.0

    return round(min(score, 10.0), 1)


def _score_to_level(score: float) -> ThreatLevel:
    if score >= 9.0:
        return ThreatLevel.CRITICAL
    if score >= 7.0:
        return ThreatLevel.HIGH
    if score >= 5.0:
        return ThreatLevel.MEDIUM
    if score >= 2.0:
        return ThreatLevel.LOW
    return ThreatLevel.INFORMATIONAL


# ══════════════════════════════════════════════════════════════════════
# TemplateReasoner — deterministic, evidence-backed report generation
# ══════════════════════════════════════════════════════════════════════

class TemplateReasoner:
    """
    Generates every InvestigationReport field from structured context data.

    This is NOT a stub or placeholder — it produces substantive reports
    by using all fields in InvestigationContext:
        - Malware names, types, descriptions
        - APT group names, countries, campaigns
        - CVE IDs, CVSS scores, CISA-KEV status
        - MITRE technique IDs and tactics
        - IOC values, tags, confidence
        - Evidence strings from the aggregation phase
    """

    # ── Executive Summary ─────────────────────────────────────────────

    def build_executive_summary(
        self,
        ctx: InvestigationContext,
        risk_score: float,
        threat_level: ThreatLevel,
    ) -> ExecutiveSummary:
        headline = self._build_headline(ctx, threat_level)
        summary  = self._build_summary(ctx, risk_score, threat_level)
        findings = self._build_key_findings(ctx)
        return ExecutiveSummary(
            headline=headline,
            threat_level=threat_level,
            summary=summary,
            key_findings=findings,
        )

    def _build_headline(self, ctx: InvestigationContext, level: ThreatLevel) -> str:
        parts: list[str] = []
        if ctx.matched_malware:
            names = ", ".join(m.name for m in ctx.matched_malware[:2])
            parts.append(f"{names} malware")
        if ctx.matched_apt_groups:
            names = ", ".join(a.name for a in ctx.matched_apt_groups[:2])
            parts.append(f"{names} threat actor activity")
        if ctx.matched_cves:
            top = ctx.matched_cves[0]
            parts.append(f"{top.cve_id} ({top.severity.upper() if top.severity else 'CVE'})")
        if ctx.matched_iocs:
            parts.append(f"{len(ctx.matched_iocs)} malicious indicator(s)")
        if ctx.matched_techniques:
            parts.append(f"{len(ctx.matched_techniques)} ATT&CK technique(s)")

        subject = " | ".join(parts) if parts else (ctx.query_summary or "Unknown indicator")
        return f"{subject} — {level.value} risk"

    def _build_summary(
        self,
        ctx: InvestigationContext,
        risk_score: float,
        level: ThreatLevel,
    ) -> str:
        sentences: list[str] = []

        # Opening sentence
        if ctx.query_summary:
            sentences.append(
                f"Investigation of '{ctx.query_summary}' returned a risk score of "
                f"{risk_score}/10 ({level.value})."
            )
        else:
            sentences.append(
                f"This investigation returned a risk score of {risk_score}/10 ({level.value})."
            )

        # Malware context
        if ctx.matched_malware:
            m = ctx.matched_malware[0]
            apt_str = f" attributed to {', '.join(m.attributed_to[:2])}" if m.attributed_to else ""
            cve_str = f" which exploits {', '.join(m.cves_exploited[:2])}" if m.cves_exploited else ""
            sentences.append(
                f"The {m.family_type or 'malware'} {m.name}{apt_str} was identified{cve_str}."
            )

        # APT context
        if ctx.matched_apt_groups:
            apt = ctx.matched_apt_groups[0]
            origin = f" ({apt.country})" if apt.country else ""
            sentences.append(
                f"The {apt.name}{origin} threat group has been associated with this activity."
            )

        # CVE context
        kev_cves = [c for c in ctx.matched_cves if c.cisa_kev]
        if kev_cves:
            ids = ", ".join(c.cve_id for c in kev_cves[:2])
            sentences.append(
                f"{ids} {'is' if len(kev_cves) == 1 else 'are'} listed in the CISA Known "
                f"Exploited Vulnerabilities catalog and require immediate patching."
            )
        elif ctx.matched_cves:
            top = ctx.matched_cves[0]
            sentences.append(
                f"{top.cve_id} (CVSS {top.cvss_score}) has exploitation status "
                f"'{top.exploitation_status or 'unknown'}'."
            )

        return " ".join(sentences)

    def _build_key_findings(self, ctx: InvestigationContext) -> list[str]:
        findings: list[str] = []

        for m in ctx.matched_malware[:3]:
            cves = f" exploits: {', '.join(m.cves_exploited[:2])}" if m.cves_exploited else ""
            findings.append(f"Malware: {m.name} ({m.family_type or 'type unknown'}){cves}")

        for a in ctx.matched_apt_groups[:3]:
            findings.append(
                f"Threat actor: {a.name} ({a.country or 'origin unknown'}) — "
                f"uses {', '.join(a.malware_used[:2]) or 'unknown malware'}"
            )

        for c in ctx.matched_cves[:3]:
            kev = " [CISA KEV — patch immediately]" if c.cisa_kev else ""
            findings.append(f"Vulnerability: {c.cve_id} CVSS {c.cvss_score}{kev}")

        for ioc in ctx.matched_iocs[:3]:
            findings.append(
                f"Malicious {ioc.ioc_type.upper()}: {ioc.value} "
                f"({', '.join(ioc.tags[:2]) or 'known-bad'})"
            )

        if ctx.matched_techniques:
            tactic_counts: dict[str, int] = {}
            for t in ctx.matched_techniques:
                tactic_counts[t.tactic] = tactic_counts.get(t.tactic, 0) + 1
            top_tactic = max(tactic_counts, key=lambda k: tactic_counts[k])
            findings.append(
                f"{len(ctx.matched_techniques)} ATT&CK technique(s) identified; "
                f"highest concentration in '{top_tactic}' tactic"
            )

        return findings or ["No significant threat indicators found."]

    # ── Threat Assessment ──────────────────────────────────────────────

    def build_threat_assessment(
        self,
        ctx: InvestigationContext,
        risk_score: float,
        threat_level: ThreatLevel,
    ) -> ThreatAssessment:
        threat_actors = [a.name for a in ctx.matched_apt_groups]
        for m in ctx.matched_malware:
            threat_actors.extend(m.attributed_to)
        threat_actors = list(dict.fromkeys(threat_actors))   # deduplicate, preserve order

        techniques = [
            f"{t.technique_id}: {t.technique_name}"
            for t in ctx.matched_techniques
        ]

        cves = [c.cve_id for c in ctx.matched_cves]
        for m in ctx.matched_malware:
            cves.extend(m.cves_exploited)
        cves = list(dict.fromkeys(cves))

        return ThreatAssessment(
            risk_score=risk_score,
            threat_level=threat_level,
            threat_actors=threat_actors,
            attack_techniques=techniques,
            exploited_cves=cves,
            ioc_count=len(ctx.matched_iocs),
            confidence=ctx.confidence_score,
        )

    # ── Evidence Summary ───────────────────────────────────────────────

    def build_evidence_summary(self, ctx: InvestigationContext) -> EvidenceSummary:
        # Try to get semantic hit count from EnrichedInvestigationContext
        semantic_hits = 0
        semantic_top = 0.0
        if hasattr(ctx, "semantic_results") and ctx.semantic_results is not None:  # type: ignore[attr-defined]
            sem = ctx.semantic_results  # type: ignore[attr-defined]
            semantic_hits = getattr(sem, "total_hits", 0)
            semantic_top = getattr(sem, "top_score", 0.0)

        sources: list[str] = list(ctx.provider_hits.keys()) if ctx.provider_hits else []

        return EvidenceSummary(
            ioc_count=len(ctx.matched_iocs),
            malware_families=[m.name for m in ctx.matched_malware],
            apt_groups=[a.name for a in ctx.matched_apt_groups],
            cve_ids=[c.cve_id for c in ctx.matched_cves],
            techniques=[f"{t.technique_id}: {t.technique_name}" for t in ctx.matched_techniques],
            semantic_hits=semantic_hits,
            semantic_top_score=semantic_top,
            sources=sources,
        )

    # ── Kill Chain ─────────────────────────────────────────────────────

    def build_kill_chain(self, ctx: InvestigationContext) -> list[KillChainPhase]:
        """
        Group techniques by tactic → kill chain phase.

        Also merges kill_chain fields from matched malware families
        when no MITRE techniques are explicitly known.
        """
        phase_map: dict[str, dict[str, Any]] = {}

        for tech in ctx.matched_techniques:
            tactic_key = tech.tactic.lower().replace(" ", "-")
            phase = _TACTIC_TO_PHASE.get(tactic_key, "Unknown")
            if phase not in phase_map:
                phase_map[phase] = {"tactics": set(), "techniques": [], "evidence": []}
            phase_map[phase]["tactics"].add(tech.tactic)
            phase_map[phase]["techniques"].append(tech.technique_id)

        # Supplement from malware kill_chain lists
        for m in ctx.matched_malware:
            for step in m.kill_chain:
                step_key = step.lower().replace(" ", "-")
                phase = _TACTIC_TO_PHASE.get(step_key, step)
                if phase not in phase_map:
                    phase_map[phase] = {"tactics": set(), "techniques": [], "evidence": []}
                phase_map[phase]["evidence"].append(f"Observed in {m.name} campaign")

        if not phase_map:
            return []

        # Sort by canonical kill chain order
        def _phase_order(p: str) -> int:
            try:
                return _PHASE_ORDER.index(p)
            except ValueError:
                return 99

        return [
            KillChainPhase(
                phase=phase,
                tactics=sorted(data["tactics"]),
                techniques=list(dict.fromkeys(data["techniques"])),
                evidence=data["evidence"],
            )
            for phase, data in sorted(phase_map.items(), key=lambda kv: _phase_order(kv[0]))
        ]

    # ── Recommendations ────────────────────────────────────────────────

    def build_recommendations(self, ctx: InvestigationContext) -> list[Recommendation]:
        recs: list[Recommendation] = []

        # ── CISA KEV patches (IMMEDIATE) ──────────────────────────────
        for c in ctx.matched_cves:
            if c.cisa_kev:
                recs.append(Recommendation(
                    priority=RecommendationPriority.IMMEDIATE,
                    category=RecommendationCategory.MITIGATION,
                    action=f"Apply vendor patch for {c.cve_id} ({c.name or c.description[:50]})",
                    rationale=(
                        f"{c.cve_id} is listed in the CISA KEV catalog "
                        f"(CVSS {c.cvss_score}) and is actively exploited."
                    ),
                ))

        # ── IOC blocking (IMMEDIATE) ───────────────────────────────────
        if ctx.matched_iocs:
            ioc_vals = ", ".join(
                i.value for i in ctx.matched_iocs[:3] if i.is_known_bad
            )
            if ioc_vals:
                recs.append(Recommendation(
                    priority=RecommendationPriority.IMMEDIATE,
                    category=RecommendationCategory.MITIGATION,
                    action=f"Block malicious indicators at perimeter: {ioc_vals}",
                    rationale=(
                        "These indicators are confirmed known-bad in threat intelligence. "
                        "Blocking prevents further communication with attacker infrastructure."
                    ),
                ))

        # ── Non-KEV CVE patches (SHORT_TERM) ─────────────────────────
        high_cves = [c for c in ctx.matched_cves if not c.cisa_kev and c.cvss_score >= 7.0]
        for c in high_cves[:2]:
            recs.append(Recommendation(
                priority=RecommendationPriority.SHORT_TERM,
                category=RecommendationCategory.MITIGATION,
                action=f"Prioritise patching {c.cve_id} (CVSS {c.cvss_score})",
                rationale=(
                    f"{c.cve_id} has a severity of {c.severity.upper() if c.severity else 'HIGH'} "
                    f"and exploitation status: {c.exploitation_status or 'unknown'}."
                ),
            ))

        # ── Malware-specific mitigations ──────────────────────────────
        for m in ctx.matched_malware[:2]:
            if "ransomware" in m.family_type.lower():
                recs.append(Recommendation(
                    priority=RecommendationPriority.IMMEDIATE,
                    category=RecommendationCategory.RESPONSE,
                    action=f"Isolate affected systems and initiate ransomware playbook for {m.name}",
                    rationale=(
                        f"{m.name} is ransomware; immediate isolation prevents lateral "
                        "spread and further encryption of shared storage."
                    ),
                ))
                recs.append(Recommendation(
                    priority=RecommendationPriority.SHORT_TERM,
                    category=RecommendationCategory.HARDENING,
                    action="Verify offline backup integrity and test restore procedures",
                    rationale="Offline backups are the primary recovery mechanism against ransomware.",
                ))
            if "trojan" in m.family_type.lower() or "rat" in m.family_type.lower():
                recs.append(Recommendation(
                    priority=RecommendationPriority.IMMEDIATE,
                    category=RecommendationCategory.DETECTION,
                    action=f"Hunt for {m.name} persistence mechanisms across the environment",
                    rationale=(
                        f"{m.name} ({m.family_type}) installs persistence to survive reboots. "
                        "Enumerate startup items, scheduled tasks, and registry run keys."
                    ),
                ))

        # ── APT-specific monitoring ───────────────────────────────────
        for a in ctx.matched_apt_groups[:2]:
            techniques_str = ", ".join(a.mitre_techniques[:3]) or "various techniques"
            recs.append(Recommendation(
                priority=RecommendationPriority.SHORT_TERM,
                category=RecommendationCategory.MONITORING,
                action=(
                    f"Enable detections for {a.name} TTPs: {techniques_str}"
                ),
                rationale=(
                    f"{a.name} ({a.country or 'nation-state'}) is a sophisticated threat actor. "
                    "Targeted detections based on their known TTPs reduce dwell time."
                ),
            ))

        # ── Always-include universal recommendations ──────────────────
        if not any(r.action.startswith("Enable multi-factor") for r in recs):
            recs.append(Recommendation(
                priority=RecommendationPriority.SHORT_TERM,
                category=RecommendationCategory.HARDENING,
                action="Enable multi-factor authentication (MFA) on all privileged accounts",
                rationale=(
                    "MFA significantly reduces the impact of credential theft, "
                    "a common initial access vector."
                ),
            ))

        if ctx.matched_techniques:
            recs.append(Recommendation(
                priority=RecommendationPriority.SHORT_TERM,
                category=RecommendationCategory.DETECTION,
                action=(
                    f"Deploy SIEM detection rules for "
                    f"{len(ctx.matched_techniques)} identified ATT&CK technique(s)"
                ),
                rationale=(
                    "Threat-specific detections aligned to observed ATT&CK techniques "
                    "maximise signal-to-noise ratio in alerts."
                ),
            ))

        recs.append(Recommendation(
            priority=RecommendationPriority.LONG_TERM,
            category=RecommendationCategory.HARDENING,
            action="Conduct tabletop exercise simulating the identified attack chain",
            rationale=(
                "Tabletop exercises validate detection coverage and incident response "
                "readiness against the specific adversary TTPs identified."
            ),
        ))

        return recs

    # ── References ─────────────────────────────────────────────────────

    def build_references(self, ctx: InvestigationContext) -> list[str]:
        refs: list[str] = []

        for c in ctx.matched_cves:
            refs.append(f"https://nvd.nist.gov/vuln/detail/{c.cve_id}")
            if c.cisa_kev:
                refs.append(
                    f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog"
                    f"?search_api_fulltext={c.cve_id}"
                )

        for a in ctx.matched_apt_groups:
            if a.mitre_group_id:
                refs.append(
                    f"https://attack.mitre.org/groups/{a.mitre_group_id}/"
                )

        for t in ctx.matched_techniques[:5]:
            refs.append(f"https://attack.mitre.org/techniques/{t.technique_id}/")

        # Deduplicate while preserving order
        return list(dict.fromkeys(refs))


# ══════════════════════════════════════════════════════════════════════
# LLM response → InvestigationReport field mapping
# ══════════════════════════════════════════════════════════════════════

def _level_from_str(s: str) -> ThreatLevel:
    try:
        return ThreatLevel(s.upper())
    except (ValueError, KeyError, AttributeError):
        return ThreatLevel.INFORMATIONAL


def _fill_from_llm(
    data: dict[str, Any],
    ctx: InvestigationContext,
    template: TemplateReasoner,
    risk_score: float,
    threat_level: ThreatLevel,
) -> tuple[ExecutiveSummary, ThreatAssessment, list[KillChainPhase], list[Recommendation]]:
    """
    Populate report sub-models from the LLM JSON response.

    Any missing or malformed field falls back to template generation.
    """
    # ── ExecutiveSummary ──────────────────────────────────────────────
    llm_level = _level_from_str(str(data.get("threat_level", "")))
    try:
        exec_summary = ExecutiveSummary(
            headline=str(data.get("headline", "")) or template._build_headline(ctx, llm_level),
            threat_level=llm_level,
            summary=str(data.get("summary", "")) or template._build_summary(ctx, risk_score, llm_level),
            key_findings=list(data.get("key_findings", [])) or template._build_key_findings(ctx),
        )
    except Exception:
        exec_summary = template.build_executive_summary(ctx, risk_score, threat_level)

    # ── ThreatAssessment ──────────────────────────────────────────────
    try:
        llm_score = float(data.get("risk_score", risk_score))
        llm_score = max(0.0, min(10.0, llm_score))
        assessment = ThreatAssessment(
            risk_score=llm_score,
            threat_level=_score_to_level(llm_score),
            threat_actors=list(data.get("threat_actors", [])),
            attack_techniques=list(data.get("attack_techniques", [])),
            exploited_cves=list(data.get("exploited_cves", [])),
            ioc_count=len(ctx.matched_iocs),
            confidence=ctx.confidence_score,
        )
    except Exception:
        assessment = template.build_threat_assessment(ctx, risk_score, threat_level)

    # ── Kill chain ────────────────────────────────────────────────────
    try:
        kc_data = data.get("kill_chain", [])
        kill_chain = [
            KillChainPhase(
                phase=str(kc.get("phase", "")),
                tactics=list(kc.get("tactics", [])),
                techniques=list(kc.get("techniques", [])),
                evidence=list(kc.get("evidence", [])),
            )
            for kc in (kc_data if isinstance(kc_data, list) else [])
            if isinstance(kc, dict) and kc.get("phase")
        ] or template.build_kill_chain(ctx)
    except Exception:
        kill_chain = template.build_kill_chain(ctx)

    # ── Recommendations ───────────────────────────────────────────────
    try:
        rec_data = data.get("recommendations", [])
        recs = [
            Recommendation(
                priority=_safe_enum(RecommendationPriority, r.get("priority", ""), RecommendationPriority.SHORT_TERM),
                category=_safe_enum(RecommendationCategory, r.get("category", ""), RecommendationCategory.MITIGATION),
                action=str(r.get("action", "Review findings")),
                rationale=str(r.get("rationale", "")),
            )
            for r in (rec_data if isinstance(rec_data, list) else [])
            if isinstance(r, dict) and r.get("action")
        ] or template.build_recommendations(ctx)
    except Exception:
        recs = template.build_recommendations(ctx)

    return exec_summary, assessment, kill_chain, recs


def _safe_enum(enum_cls, value: str, default):
    try:
        return enum_cls(str(value).upper())
    except (ValueError, KeyError, AttributeError):
        return default


# ══════════════════════════════════════════════════════════════════════
# ReportBuilder — public entry point
# ══════════════════════════════════════════════════════════════════════

class ReportBuilder:
    """
    Generates an InvestigationReport from an InvestigationContext.

    Injection-friendly: llm_reasoner and template_reasoner can be swapped
    for testing or alternative implementations.

    Usage:
        builder = ReportBuilder()
        report  = builder.build(context)

    With mock LLM (testing):
        builder = ReportBuilder(llm_reasoner=MockLLMReasoner())
        report  = builder.build(context)
    """

    def __init__(
        self,
        llm_reasoner:     LLMReasoner | None = None,
        template_reasoner: TemplateReasoner | None = None,
        use_llm: bool = True,
    ) -> None:
        """
        Args:
            llm_reasoner:     Inject a custom LLMReasoner (default: production Ollama client)
            template_reasoner: Inject a custom TemplateReasoner (default: TemplateReasoner())
            use_llm:          Set False to skip LLM entirely (always use template)
        """
        self._llm_reasoner     = llm_reasoner
        self._template_reasoner = template_reasoner or TemplateReasoner()
        self._use_llm          = use_llm

    @property
    def llm_reasoner(self) -> LLMReasoner:
        if self._llm_reasoner is None:
            self._llm_reasoner = LLMReasoner()
        return self._llm_reasoner

    def build(self, ctx: InvestigationContext) -> InvestigationReport:
        """
        Build a complete InvestigationReport from an InvestigationContext.

        Args:
            ctx: InvestigationContext OR EnrichedInvestigationContext.

        Returns:
            InvestigationReport — all fields populated, JSON-serialisable.
        """
        logger.info(
            "Building report for request %s [%s]",
            ctx.request_id,
            ctx.input_type,
        )

        # ── Compute universal scores (needed by both paths) ────────────
        risk_score   = _compute_risk_score(ctx)
        threat_level = _score_to_level(risk_score)

        reasoning_method = ReasoningMethod.TEMPLATE
        llm_model        = ""
        llm_output: ReasoningOutput | None = None

        # ── Try LLM path ──────────────────────────────────────────────
        if self._use_llm:
            try:
                system, user = PromptBuilder.build(ctx)
                llm_output   = self.llm_reasoner.reason(system, user)
                if llm_output.parsed:
                    reasoning_method = ReasoningMethod.OLLAMA
                    llm_model        = llm_output.model
            except Exception:
                logger.exception("LLM reasoning step failed — falling back to template")
                llm_output = None

        # ── Build sub-models (LLM-assisted or pure template) ──────────
        tmpl = self._template_reasoner

        if llm_output and llm_output.parsed and llm_output.data:
            exec_summary, assessment, kill_chain, recommendations = _fill_from_llm(
                llm_output.data, ctx, tmpl, risk_score, threat_level
            )
        else:
            exec_summary   = tmpl.build_executive_summary(ctx, risk_score, threat_level)
            assessment     = tmpl.build_threat_assessment(ctx, risk_score, threat_level)
            kill_chain     = tmpl.build_kill_chain(ctx)
            recommendations = tmpl.build_recommendations(ctx)

        evidence  = tmpl.build_evidence_summary(ctx)
        references = tmpl.build_references(ctx)

        # ── Extract input value from request_id or query_summary ──────
        input_value = ctx.query_summary or ctx.request_id

        report = InvestigationReport(
            request_id=ctx.request_id,
            input_value=input_value,
            input_type=ctx.input_type,
            executive_summary=exec_summary,
            threat_assessment=assessment,
            evidence_summary=evidence,
            kill_chain=kill_chain,
            recommendations=recommendations,
            references=references,
            reasoning_method=reasoning_method,
            llm_model=llm_model,
            confidence_score=ctx.confidence_score,
        )

        logger.info(
            "Report built: %s | risk=%s/10 | method=%s | recs=%d | kc_phases=%d",
            report.executive_summary.threat_level.value,
            risk_score,
            reasoning_method.value,
            len(recommendations),
            len(kill_chain),
        )
        return report


# ══════════════════════════════════════════════════════════════════════
# Module-level convenience function
# ══════════════════════════════════════════════════════════════════════

def build_report(
    ctx: InvestigationContext,
    use_llm: bool = True,
) -> InvestigationReport:
    """
    Convenience function — builds an InvestigationReport from a context.

    Args:
        ctx:     InvestigationContext or EnrichedInvestigationContext.
        use_llm: If False, skip Ollama and use the template reasoner only.

    Returns:
        InvestigationReport
    """
    return ReportBuilder(use_llm=use_llm).build(ctx)
