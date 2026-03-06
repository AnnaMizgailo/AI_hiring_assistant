import json
import os
from typing import Any, Dict, List, Optional

import requests

from rag_text_utils import clean_text, keyword_coverage


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
DEFAULT_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")


def ollama_generate(model: str, prompt: str, temperature: float = 0.2) -> str:
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


def simple_keywords_from_job_text(job_text: str) -> List[str]:
    import re

    tokens = re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9\+\#\.\-/]{1,30}", job_text)

    stop = {
        "job",
        "title",
        "location",
        "employment",
        "type",
        "company",
        "description",
        "responsibilities",
        "requirements",
        "nice",
        "have",
        "salary",
        "keywords",
        "remote",
        "команда",
        "будет",
        "работа",
        "важно",
        "ищем",
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
    job_description: str,
    requirements: List[str],
    evidence_chunks: List[Dict[str, Any]],
) -> str:
    reqs = "\n".join([f"- {r}" for r in requirements]) if requirements else "- No explicit requirements"
    ev = "\n\n".join(
        [
            f"- Evidence (score={float(ch.get('score') or 0.0):.3f}):\n{clean_text(str(ch.get('text') or ''))}"
            for ch in evidence_chunks
        ]
    )

    job_text = f"{job_title}\n\n{job_description}\n\n" + "\n".join(requirements)
    job_keywords = simple_keywords_from_job_text(job_text)
    kw_cov = keyword_coverage(resume_text, job_keywords)
    kw_line = ", ".join(job_keywords)

    return f"""
You are an expert recruiter and ATS evaluator.

TASK:
Score how well a candidate matches ONE job.

RULES:
- Use ONLY the provided resume and retrieved evidence.
- Be realistic and strict.
- Output MUST be valid JSON only.
- Score must be an integer from 0 to 100.
- missing_requirements must contain only items from the REQUIREMENTS list below.
- rationale must be short and factual.
- evidence_snippets must be an array of objects with fields text and why.
- If evidence is weak or indirect, keep the score lower.
- Consider keyword coverage in your judgement.

JOB TITLE:
{job_title}

JOB DESCRIPTION:
{job_description}

REQUIREMENTS:
{reqs}

JOB KEYWORDS:
{kw_line}

KEYWORD COVERAGE:
{kw_cov}%

RESUME:
{resume_text}

RETRIEVED RESUME EVIDENCE:
{ev}

OUTPUT JSON SCHEMA:
{{
  "score": 0,
  "rationale": "string",
  "missing_requirements": ["string"],
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
    job_description: str,
    requirements: List[str],
    evidence_chunks: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    prompt = build_scoring_prompt(
        resume_text=resume_text,
        job_title=job_title,
        job_description=job_description,
        requirements=requirements,
        evidence_chunks=evidence_chunks,
    )

    raw = ollama_generate(model=model or DEFAULT_MODEL, prompt=prompt, temperature=0.2)

    try:
        data = safe_json_loads(raw)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    if "score" not in data or "rationale" not in data or "missing_requirements" not in data:
        return None

    try:
        score = int(data.get("score"))
    except Exception:
        return None

    rationale = str(data.get("rationale") or "").strip()
    missing = data.get("missing_requirements") or []
    evidence = data.get("evidence_snippets") or []

    if not isinstance(missing, list):
        missing = []
    if not isinstance(evidence, list):
        evidence = []

    missing = [str(x).strip() for x in missing if str(x).strip()]
    evidence_out: List[Dict[str, Any]] = []
    for item in evidence[:12]:
        if isinstance(item, dict):
            txt = str(item.get("text") or "").strip()
            why = str(item.get("why") or "").strip()
            if txt:
                evidence_out.append({"text": txt, "why": why})

    return {
        "score": max(0, min(100, score)),
        "rationale": rationale or "No rationale provided",
        "missing_requirements": missing,
        "evidence_snippets": evidence_out,
    }