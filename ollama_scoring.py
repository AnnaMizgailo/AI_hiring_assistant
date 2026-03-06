import json
import os
import re
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
        return json.loads(text[start:end + 1])
    return json.loads(text)


def simple_keywords_from_text(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9\+\#\.\-/]{1,30}", text)
    stop = {
        "job", "title", "location", "employment", "type", "company", "description",
        "responsibilities", "requirements", "nice", "have", "salary", "keywords",
        "remote", "команда", "будет", "работа", "важно", "ищем", "опыт", "skills"
    }
    out: List[str] = []
    seen = set()
    for t in tokens:
        tl = t.lower()
        if tl in stop:
            continue
        if len(t) < 2:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out[:80]


def build_scoring_prompt(
    resume_text: str,
    job_title: str,
    requirement_groups: Dict[str, List[str]],
    evidence_chunks: List[Dict[str, Any]],
    keyword_coverage_percent: float,
    evidence_keywords: List[str],
) -> str:
    must = requirement_groups.get("must_have", []) or []
    nice = requirement_groups.get("nice_to_have", []) or []
    other = requirement_groups.get("other", []) or []
    all_reqs = requirement_groups.get("all", []) or []

    def block(name: str, items: List[str]) -> str:
        if not items:
            return f"{name}:\n- none"
        return name + ":\n" + "\n".join([f"- {x}" for x in items])

    ev = "\n\n".join(
        [
            f"- Evidence {i + 1} (score={float(ch.get('score') or 0.0):.3f}, query={clean_text(str(ch.get('query') or ''))}):\n{clean_text(str(ch.get('text') or ''))}"
            for i, ch in enumerate(evidence_chunks[:8])
        ]
    ) or "- No evidence retrieved"

    kw_line = ", ".join(evidence_keywords)
    req_line = "\n".join([f"- {x}" for x in all_reqs]) if all_reqs else "- none"

    return f"""
You are an expert recruiter and ATS evaluator.

TASK:
Review one candidate resume against one job using grounded evidence.

RULES:
- Use ONLY the provided resume and retrieved evidence.
- Be realistic and strict.
- Output MUST be valid JSON only.
- You are NOT the primary scoring engine.
- Do not return a final score.
- Return a score_adjustment integer only in range [-5, 5].
- negative adjustment should be rare and used only when evidence clearly contradicts the current fit.
- If evidence is mixed or partial, keep score_adjustment near 0.
- missing_requirements must contain only items from ALLOWED REQUIREMENTS below.
- Use EVIDENCE KEYWORDS as the primary grounding source.
- Do not invent unrelated tools.
- evidence_snippets must reuse exact text from RETRIEVED RESUME EVIDENCE.

JOB TITLE:
{job_title}

REQUIREMENT GROUPS:
{block('MUST_HAVE', must)}

{block('NICE_TO_HAVE', nice)}

{block('OTHER_REQUIREMENTS', other)}

ALLOWED REQUIREMENTS:
{req_line}

EVIDENCE KEYWORDS:
{kw_line}

KEYWORD COVERAGE:
{keyword_coverage_percent}%

RESUME:
{resume_text[:5000]}

RETRIEVED RESUME EVIDENCE:
{ev}

OUTPUT JSON SCHEMA:
{{
  "rationale": "string",
  "missing_requirements": ["string"],
  "score_adjustment": 0,
  "evidence_snippets": [
    {{
      "text": "string",
      "why": "string"
    }}
  ]
}}
""".strip()


def llm_score_application(
    resume_text: str,
    job_title: str,
    requirement_groups: Dict[str, List[str]],
    evidence_chunks: List[Dict[str, Any]],
    keyword_coverage_percent: float,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    evidence_text = "\n".join([clean_text(str(ch.get("text") or "")) for ch in evidence_chunks])
    evidence_keywords = simple_keywords_from_text(evidence_text)
    prompt = build_scoring_prompt(
        resume_text=resume_text,
        job_title=job_title,
        requirement_groups=requirement_groups,
        evidence_chunks=evidence_chunks,
        keyword_coverage_percent=keyword_coverage_percent,
        evidence_keywords=evidence_keywords,
    )
    raw = ollama_generate(model=model or DEFAULT_MODEL, prompt=prompt, temperature=0.15)
    try:
        data = safe_json_loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    rationale = str(data.get("rationale") or "").strip()
    missing = data.get("missing_requirements") or []
    evidence = data.get("evidence_snippets") or []
    score_adjustment = data.get("score_adjustment", 0)

    if not isinstance(missing, list):
        missing = []
    if not isinstance(evidence, list):
        evidence = []

    try:
        score_adjustment = int(score_adjustment)
    except Exception:
        score_adjustment = 0
    score_adjustment = max(-5, min(5, score_adjustment))

    missing = [str(x).strip() for x in missing if str(x).strip()]
    evidence_out: List[Dict[str, Any]] = []
    for item in evidence[:12]:
        if isinstance(item, dict):
            txt = clean_text(str(item.get("text") or ""))
            why = clean_text(str(item.get("why") or ""))
            if txt:
                evidence_out.append({"text": txt, "why": why})

    if not rationale:
        return None

    return {
        "rationale": rationale,
        "missing_requirements": missing,
        "score_adjustment": score_adjustment,
        "evidence_snippets": evidence_out,
        "evidence_keywords": evidence_keywords,
    }
