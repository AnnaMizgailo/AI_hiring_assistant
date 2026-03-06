import json
import os
from typing import Any, Dict, List, Optional

import requests

from rag_text_utils import clean_text


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
DEFAULT_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")


def ollama_generate(model: str, prompt: str, temperature: float = 0.15) -> str:
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
        candidate = text[start:end + 1]
        return json.loads(candidate)
    return json.loads(text)


def _render_group(title: str, items: List[str]) -> str:
    if not items:
        return f"{title}:\n- none"
    return title + ":\n" + "\n".join([f"- {x}" for x in items])


def build_scoring_prompt(
    job_title: str,
    job_description: str,
    requirement_groups: Dict[str, List[str]],
    evidence_chunks: List[Dict[str, Any]],
) -> str:
    must_have = requirement_groups.get("must_have", []) or []
    nice_to_have = requirement_groups.get("nice_to_have", []) or []
    other = requirement_groups.get("other", []) or []
    all_requirements = requirement_groups.get("all", []) or []

    evidence_block = "\n\n".join(
        [
            f"EVIDENCE {i + 1} | similarity={float(ch.get('score') or 0.0):.3f} | query={clean_text(str(ch.get('query') or ''))}\n{clean_text(str(ch.get('text') or ''))}"
            for i, ch in enumerate(evidence_chunks[:8])
            if clean_text(str(ch.get("text") or ""))
        ]
    ) or "No evidence retrieved"

    allowed_requirements = "\n".join([f"- {x}" for x in all_requirements]) if all_requirements else "- none"

    return f"""
You are an expert ATS reviewer.

TASK:
Review retrieved resume evidence for one candidate and one job.

IMPORTANT:
- You are not the primary scoring engine.
- A deterministic scorer already computed the base score.
- Your job is only to:
  1) write a short grounded rationale
  2) suggest missing requirements from the allowed list only
  3) suggest a small score_adjustment between -10 and 10
  4) choose the evidence snippets that best support your reasoning

RULES:
- Use only the evidence below.
- Do not invent skills or experience.
- Do not copy the prompt wrapper into evidence text.
- Do not use one-word snippets like B1, Docker, REST as evidence.
- Keep score_adjustment close to 0 unless the evidence is clearly much better or much worse than the base score would imply.
- Output valid JSON only.

JOB TITLE:
{clean_text(job_title)}

JOB DESCRIPTION:
{clean_text(job_description)[:1800]}

{_render_group("MUST_HAVE", must_have)}

{_render_group("NICE_TO_HAVE", nice_to_have)}

{_render_group("OTHER_REQUIREMENTS", other)}

ALLOWED ITEMS FOR missing_requirements:
{allowed_requirements}

RETRIEVED RESUME EVIDENCE:
{evidence_block}

OUTPUT JSON SCHEMA:
{{
  "rationale": "string",
  "missing_requirements": ["string"],
  "score_adjustment": 0,
  "evidence_indices": [1, 2]
}}
""".strip()


def llm_score_application(
    job_title: str,
    job_description: str,
    requirement_groups: Dict[str, List[str]],
    evidence_chunks: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    prompt = build_scoring_prompt(
        job_title=job_title,
        job_description=job_description,
        requirement_groups=requirement_groups,
        evidence_chunks=evidence_chunks,
    )

    raw = ollama_generate(model=model or DEFAULT_MODEL, prompt=prompt, temperature=0.15)

    try:
        data = safe_json_loads(raw)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    rationale = clean_text(str(data.get("rationale") or ""))
    missing_requirements = data.get("missing_requirements") or []
    evidence_indices = data.get("evidence_indices") or []

    try:
        score_adjustment = int(data.get("score_adjustment", 0))
    except Exception:
        score_adjustment = 0

    score_adjustment = max(-10, min(10, score_adjustment))

    if not isinstance(missing_requirements, list):
        missing_requirements = []
    if not isinstance(evidence_indices, list):
        evidence_indices = []

    normalized_missing = [clean_text(str(x)) for x in missing_requirements if clean_text(str(x))]

    normalized_indices: List[int] = []
    for idx in evidence_indices:
        try:
            iv = int(idx)
        except Exception:
            continue
        if iv < 1 or iv > len(evidence_chunks):
            continue
        if iv not in normalized_indices:
            normalized_indices.append(iv)

    if not rationale:
        return None

    return {
        "rationale": rationale,
        "missing_requirements": normalized_missing,
        "score_adjustment": score_adjustment,
        "evidence_indices": normalized_indices,
    }
