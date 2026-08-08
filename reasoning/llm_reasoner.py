"""
reasoning/llm_reasoner.py — Local LLM Reasoning via Ollama.

Optimised for Qwen3:8b running on Ollama.

Responsibilities:
    1. Check if Ollama is available (GET /api/tags on localhost:11434)
    2. If available → send prompt, parse JSON response → ReasoningOutput
    3. If unavailable / response unparseable → set method="template"
       (report_builder.py then uses the deterministic template path)

Ollama is the ONLY LLM backend supported.  No OpenAI.  No Anthropic.

Design:
    - OllamaClient is injected into LLMReasoner → fully testable via mock
    - A 5-second availability check avoids long waits on cold start
    - JSON is requested via Ollama's format="json" parameter
    - Qwen3 models output <think>…</think> chain-of-thought blocks;
      _extract_json() strips these before parsing
    - System prompt explicitly forbids <think>, markdown, and prose
    - If the model produces invalid JSON we log a warning and fall back
    - All network calls use httpx (already in requirements.txt)

Supported Ollama models (preference order):
    qwen3:8b        (default — best quality/speed for local SOC use)
    qwen3           (any other qwen3 variant)
    llama3.2        (~2 GB, fast on modern CPUs)
    mistral         (7B, high quality)
    gemma2          (small, excellent reasoning)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL      = "http://localhost:11434"
DEFAULT_MODEL        = "qwen3:8b"
AVAILABILITY_TIMEOUT = 5    # seconds — fast check
GENERATE_TIMEOUT     = 90   # seconds — allow for slower models

# Preference order for best_available_model()
_MODEL_PREFERENCE = [
    "qwen3:8b",
    "qwen3",
    "llama3.2",
    "llama3.2:1b",
    "mistral",
    "gemma2:2b",
    "gemma2",
    "llama3",
]

# Regex to strip Qwen3 chain-of-thought blocks (greedy, dotall)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════
# ReasoningOutput — what LLMReasoner returns to report_builder
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ReasoningOutput:
    """
    Output from the LLM Reasoning step.

    parsed    — True if the LLM returned valid JSON that we could use
    method    — "ollama" if an LLM was used, "template" if not
    model     — Ollama model name (empty string if template)
    data      — parsed JSON dict from the LLM response (may be empty)
    raw_text  — raw text returned by the LLM (for debugging)
    error     — error message if something went wrong
    """
    parsed:   bool             = False
    method:   str              = "template"
    model:    str              = ""
    data:     dict[str, Any]   = field(default_factory=dict)
    raw_text: str              = ""
    error:    str              = ""


# ══════════════════════════════════════════════════════════════════════
# OllamaClient — thin wrapper around the Ollama REST API
# ══════════════════════════════════════════════════════════════════════

class OllamaClient:
    """
    Minimal Ollama REST API client.

    All external calls are in this class so they can be mocked in tests.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        """
        Return True if the Ollama server is reachable and has at least one model.

        Uses a short timeout so this never blocks the pipeline.
        """
        try:
            import httpx
            resp = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=AVAILABILITY_TIMEOUT,
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = data.get("models", [])
            return len(models) > 0
        except Exception as exc:
            logger.debug("Ollama not available: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Return names of all models available in Ollama."""
        try:
            import httpx
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=AVAILABILITY_TIMEOUT)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def best_available_model(self) -> str:
        """
        Return the best available model by preference order, or the configured default.

        Preference: qwen3:8b > qwen3 > llama3.2 > mistral > gemma2 > anything else
        """
        available = self.list_models()
        logger.debug("Available Ollama models: %s", available)
        if not available:
            logger.debug("No models found — using configured default: %s", self.model)
            return self.model
        for pref in _MODEL_PREFERENCE:
            for m in available:
                if m.startswith(pref):
                    logger.debug("Selected model '%s' (matched preference '%s')", m, pref)
                    return m
        # Fallback: first available model
        logger.debug("No preferred model found — using first available: %s", available[0])
        return available[0]

    def generate(self, system: str, user: str, model: str | None = None) -> str | None:
        """
        Call Ollama /api/generate and return the raw text response.

        Uses:
          - Qwen-compatible system/user prompt format (no [INST] tags)
          - format="json" to request structured JSON output
          - /no_think directive + explicit system instruction to suppress
            chain-of-thought and return ONLY JSON

        Returns None on any error.
        """
        _model = model or self.model
        logger.debug("Calling Ollama model: %s", _model)

        # Qwen3-compatible prompt: plain system + user sections.
        # A JSON-enforcement header is prepended to the caller's system prompt so
        # the constraint is always present regardless of what prompts.py sends.
        # The /no_think suffix on the user turn suppresses <think> blocks
        # for models that respect it (qwen3). For others it is ignored.
        _json_enforcement = (
            "CRITICAL INSTRUCTION: You MUST respond with ONLY a valid JSON object. "
            "Do NOT output any <think> blocks, chain-of-thought, reasoning steps, "
            "markdown code fences, prose explanations, or any text outside the JSON "
            "object. Your entire response must start with '{' and end with '}'."
        )
        prompt = (
            f"System:\n{_json_enforcement}\n\n{system}\n\n"
            f"User:\n{user} /no_think"
        )

        try:
            import httpx
            payload: dict[str, Any] = {
                "model":  _model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,   # low temp = more deterministic JSON
                    "num_predict": 1024,
                    "stop": ["</s>", "<|im_end|>"],  # common Qwen/Llama stop tokens
                },
            }
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=GENERATE_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning("Ollama /api/generate returned %d", resp.status_code)
                return None
            result = resp.json()
            raw = result.get("response", "")
            logger.debug("Raw Ollama response (%d chars): %s", len(raw), raw[:500])
            return raw
        except Exception as exc:
            logger.warning("Ollama generate failed: %s", exc)
            return None


# ══════════════════════════════════════════════════════════════════════
# LLMReasoner — orchestrates availability check + generate + parse
# ══════════════════════════════════════════════════════════════════════

class LLMReasoner:
    """
    Orchestrates LLM reasoning against an Ollama backend.

    Usage (production):
        reasoner = LLMReasoner()
        output   = reasoner.reason(system_prompt, user_prompt)
        if output.parsed:
            # use output.data to populate InvestigationReport
        else:
            # output.method == "template" → use TemplateReasoner

    Usage (tests):
        mock_client = MockOllamaClient(...)
        reasoner    = LLMReasoner(client=mock_client)
    """

    def __init__(self, client: OllamaClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> OllamaClient:
        if self._client is None:
            self._client = OllamaClient()
        return self._client

    def reason(self, system: str, user: str) -> ReasoningOutput:
        """
        Try to call Ollama and parse the result.

        Returns:
            ReasoningOutput with method="ollama" if successful,
            method="template" if Ollama is unavailable or response is invalid.
        """
        if not self.client.is_available():
            logger.info("Ollama not available — using template reasoning.")
            return ReasoningOutput(method="template", error="Ollama server not reachable")

        model = self.client.best_available_model()
        logger.info("Selected Ollama model: '%s'", model)

        raw = self.client.generate(system, user, model=model)
        if not raw:
            logger.info("Ollama returned empty response — falling back to template.")
            return ReasoningOutput(
                method="template",
                model=model,
                error="Ollama returned empty response",
            )

        logger.debug("Raw Ollama response before parsing: %s", raw[:500])

        # ── Try to parse JSON ─────────────────────────────────────────
        try:
            data = _extract_json(raw)
            logger.info(
                "Ollama response parsed successfully: model=%s keys=%d",
                model, len(data),
            )
            return ReasoningOutput(
                parsed=True,
                method="ollama",
                model=model,
                data=data,
                raw_text=raw,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Ollama response is not valid JSON (model=%s error=%s) — "
                "falling back to template. Raw snippet: %s",
                model, exc, raw[:300],
            )
            return ReasoningOutput(
                parsed=False,
                method="template",
                model=model,
                raw_text=raw,
                error=f"JSON parse error: {exc}",
            )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract the first JSON object from a string.

    Handles (in order):
        1. <think>…</think> chain-of-thought blocks (Qwen3)
        2. Markdown code fences (```json … ``` or ``` … ```)
        3. Leading / trailing prose around the JSON object
        4. Raw JSON
    """
    # 1. Strip <think>…</think> blocks (Qwen3 chain-of-thought)
    text = _THINK_RE.sub("", text).strip()

    # 2. Strip markdown fences
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            if "```" in text:
                text = text[: text.index("```")]
            text = text.strip()
            break

    # 3. Find the first { … } JSON object (ignore leading prose)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    # Walk forward to find the matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError("Unclosed JSON object in response")
