import json
import os
from typing import Any, Dict, List, Optional

import requests

from rag_text_utils import clean_text


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
DEFAULT_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")


def ollama_generate(model: str, prompt: str, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(DEFAULT_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


def _render_list(items: List[str]) -> str:
    if not items:
        return "- none"
    return "\n".join([f"- {x}" for x in items])


def build_scoring_prompt(
    job_title: str,
    deterministic_facts: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
    max_abs_adjustment: int,
) -> str:
    must_have = deterministic_facts.get("must_have", []) or []
    nice_to_have = deterministic_facts.get("nice_to_have", []) or []
    must_hits = deterministic_facts.get("must_hits", []) or []
    must_missing = deterministic_facts.get("must_missing", []) or []
    nice_hits = deterministic_facts.get("nice_hits", []) or []
    nice_missing = deterministic_facts.get("nice_missing", []) or []
    other_hits = deterministic_facts.get("other_hits", []) or []
    other_missing = deterministic_facts.get("other_missing", []) or []
    deterministic_score = int(deterministic_facts.get("deterministic_score") or 0)
    keyword_coverage_percent = float(deterministic_facts.get("keyword_coverage_percent") or 0.0)

    evidence = []
    for i, item in enumerate(evidence_chunks[:6], start=1):
        evidence.append(
            f"Evidence {i}:\n"
            f"text: {clean_text(str(item.get('text') or ''))}\n"
            f"query: {clean_text(str(item.get('query') or ''))}\n"
            f"score: {float(item.get('score') or 0.0):.3f}"
        )
    evidence_block = "\n\n".join(evidence) if evidence else "No evidence"

    return f"""
You explain resume scoring.

You are NOT the primary scorer.
The deterministic scorer is the source of truth for matched and missing requirements.

Your job:
1. Write a short factual rationale using the deterministic facts and evidence.
2. Suggest a small integer score_adjustment between {-max_abs_adjustment} and {max_abs_adjustment}.
3. Optionally highlight a subset of the already-missing requirements.

Rules:
- Do not say a must-have requirement is met if it appears in MUST_HAVE_MISSING.
- Do not invent skills.
- Use only the evidence and facts below.
- If deterministic facts say must-have coverage is weak, do not write that all must-have requirements are met.
- focus_missing must be a subset of ALL_MISSING.
- Return valid JSON only.

JOB_TITLE:
{clean_text(job_title)}

MUST_HAVE:
{_render_list(must_have)}

NICE_TO_HAVE:
{_render_list(nice_to_have)}

MUST_HAVE_HITS:
{_render_list(must_hits)}

MUST_HAVE_MISSING:
{_render_list(must_missing)}

NICE_TO_HAVE_HITS:
{_render_list(nice_hits)}

NICE_TO_HAVE_MISSING:
{_render_list(nice_missing)}

OTHER_HITS:
{_render_list(other_hits)}

OTHER_MISSING:
{_render_list(other_missing)}

ALL_MISSING:
{_render_list((must_missing + nice_missing + other_missing))}

DETERMINISTIC_SCORE:
{deterministic_score}

KEYWORD_COVERAGE_PERCENT:
{keyword_coverage_percent}

EVIDENCE:
{evidence_block}

Return JSON:
{{
  "rationale": "string",
  "score_adjustment": 0,
  "focus_missing": ["string"]
}}
""".strip()


def llm_score_application(
    job_title: str,
    deterministic_facts: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
    model: Optional[str] = None,
    max_abs_adjustment: int = 5,
) -> Optional[Dict[str, Any]]:
    prompt = build_scoring_prompt(
        job_title=job_title,
        deterministic_facts=deterministic_facts,
        evidence_chunks=evidence_chunks,
        max_abs_adjustment=max_abs_adjustment,
    )

    raw = ollama_generate(model=model or DEFAULT_MODEL, prompt=prompt, temperature=0.1)

    try:
        data = safe_json_loads(raw)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    rationale = clean_text(str(data.get("rationale") or ""))
    focus_missing = data.get("focus_missing") or []
    score_adjustment = data.get("score_adjustment", 0)

    try:
        score_adjustment = int(score_adjustment)
    except Exception:
        score_adjustment = 0

    score_adjustment = max(-max_abs_adjustment, min(max_abs_adjustment, score_adjustment))

    if not isinstance(focus_missing, list):
        focus_missing = []

    focus_missing = [clean_text(str(x or "")) for x in focus_missing if clean_text(str(x or ""))]

    if not rationale:
        return None

    return {
        "rationale": rationale,
        "score_adjustment": score_adjustment,
        "focus_missing": focus_missing,
    }