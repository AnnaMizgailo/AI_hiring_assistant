from typing import Any, Dict, List, Tuple, Optional
import asyncio
import datetime
import re
import uuid

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Application, Document, Job, Profile, RagRun
from ollama_scoring import llm_score_application
from rag_text_utils import clean_text, keyword_coverage, normalize, split_into_chunks


_CONTAINER_KEYS = {
    "must_have",
    "must_haves",
    "nice_to_have",
    "nice_to_haves",
    "requirements",
    "requirement",
    "responsibilities",
    "responsibility",
    "languages",
    "language",
    "experience",
    "education",
    "skills",
    "tech_stack",
    "stack",
    "tools",
    "summary",
}

_NOISE_PATTERNS = [
    r"^b[12c]$",
    r"^a[12]$",
    r"^c2$",
    r"^native$",
    r"^upper\-intermediate$",
    r"^intermediate$",
    r"^advanced$",
    r"^город[:\s]",
    r"^email[:\s]",
    r"^телефон[:\s]",
    r"^phone[:\s]",
    r"^telegram[:\s]",
    r"^location[:\s]",
    r"^evidence\s+\d+",
]


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _extract_strings_only(obj: Any, max_items: int = 400) -> List[str]:
    out: List[str] = []

    def rec(x: Any) -> None:
        nonlocal out
        if len(out) >= max_items:
            return
        if x is None:
            return
        if isinstance(x, str):
            s = clean_text(x)
            if s:
                out.append(s)
            return
        if isinstance(x, (int, float, bool)):
            return
        if isinstance(x, dict):
            for _, v in x.items():
                if len(out) >= max_items:
                    break
                rec(v)
            return
        if isinstance(x, list):
            for it in x:
                if len(out) >= max_items:
                    break
                rec(it)

    rec(obj)

    uniq: List[str] = []
    seen = set()
    for s in out:
        sl = s.lower()
        if sl in seen:
            continue
        seen.add(sl)
        uniq.append(s)
    return uniq


def _clean_requirement_list(items: List[str], limit: int = 80) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = clean_text(item)
        if not value:
            continue
        key = normalize(value)
        if key in _CONTAINER_KEYS:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out[:limit]


def _extract_requirement_groups(requirements_json: Dict[str, Any]) -> Dict[str, List[str]]:
    rj = requirements_json or {}

    must_have = _clean_requirement_list(
        _extract_strings_only(rj.get("must_have") or rj.get("must_haves") or [], max_items=200)
    )
    nice_to_have = _clean_requirement_list(
        _extract_strings_only(rj.get("nice_to_have") or rj.get("nice_to_haves") or [], max_items=200)
    )
    other = _clean_requirement_list(
        _extract_strings_only(rj.get("requirements") or [], max_items=200)
        + _extract_strings_only(rj.get("experience") or [], max_items=100)
        + _extract_strings_only(rj.get("languages") or [], max_items=100)
    )

    if not must_have and not nice_to_have and not other:
        must_have = _clean_requirement_list(_extract_strings_only(rj, max_items=300), limit=120)

    all_reqs = _clean_requirement_list(must_have + nice_to_have + other, limit=120)

    return {
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "other": other,
        "all": all_reqs,
    }


def _extract_requirements(requirements_json: Dict[str, Any]) -> List[str]:
    return _extract_requirement_groups(requirements_json).get("all", [])


def _profile_skill_strings(profile_json: Dict[str, Any]) -> List[str]:
    profile_json = profile_json or {}
    items: List[str] = []
    for key in ["skills", "technologies", "stack", "tools", "languages", "frameworks", "certifications", "summary", "projects", "experience"]:
        if key in profile_json:
            items.extend(_extract_strings_only(profile_json.get(key), max_items=250))
    if not items:
        items = _extract_strings_only(profile_json, max_items=500)

    uniq: List[str] = []
    seen = set()
    for s in items:
        sl = normalize(s)
        if sl in seen:
            continue
        seen.add(sl)
        uniq.append(s)
    return uniq


def _canonical_text(text: str) -> str:
    t = normalize(text)
    replacements = [
        ("typescript", "ts"),
        ("type script", "ts"),
        ("next.js", "nextjs"),
        ("next js", "nextjs"),
        ("nextjs", "nextjs"),
        ("postgresql", "postgres"),
        ("postgre sql", "postgres"),
        ("ci/cd", "cicd"),
        ("ci cd", "cicd"),
        ("a/b testing", "abtesting"),
        ("ab testing", "abtesting"),
        ("a b testing", "abtesting"),
        ("rest api", "rest"),
        ("product management", "productmanager"),
        ("product manager", "productmanager"),
        ("jobs to be done", "jtbd"),
        ("job to be done", "jtbd"),
        ("test design", "testdesign"),
        ("code review", "codereview"),
        ("code reviews", "codereview"),
    ]
    for src, dst in replacements:
        t = t.replace(src, dst)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _query_terms(text: str) -> List[str]:
    parts = re.findall(r"[A-Za-zА-Яа-я0-9\+#\./-]{2,}", _canonical_text(text))
    out: List[str] = []
    seen = set()
    for p in parts:
        if len(p) < 3:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out[:8]


def _sentence_split(text: str) -> List[str]:
    t = clean_text(text)
    if not t:
        return []
    parts = re.split(r"(?<=[\.!\?])\s+|\n+", t)
    return [clean_text(x) for x in parts if clean_text(x)]


def _is_noise_text(text: str) -> bool:
    value = normalize(text)
    if not value:
        return True
    if len(value) < 12:
        return True
    if len(re.findall(r"\w+", value)) < 3:
        return True
    for pat in _NOISE_PATTERNS:
        if re.search(pat, value):
            return True
    if value.startswith("evidence "):
        return True
    return False


def _tech_density(text: str) -> float:
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9\+#\./-]{2,}", text or "")
    if not tokens:
        return 0.0
    tech_words = {
        "python", "django", "fastapi", "flask", "postgres", "postgresql", "redis", "docker", "kubernetes",
        "react", "typescript", "nextjs", "next", "sql", "pandas", "tableau", "airflow", "spark", "mlflow",
        "playwright", "selenium", "rest", "graphql", "git", "cicd", "jtbd", "figma", "scrum", "agile",
        "celery", "kafka", "grpc", "java", "javascript", "typescript", "html", "css", "aws", "gcp",
    }
    hits = 0
    for tok in tokens:
        tl = normalize(tok)
        if tl in tech_words or any(ch.isdigit() for ch in tl) or "+" in tl or "#" in tl or "/" in tl or "." in tl:
            hits += 1
    return min(1.0, hits / max(1, len(tokens)))


def _best_display_snippet(chunk_text: str, query: str, max_len: int = 320) -> str:
    sentences = _sentence_split(chunk_text)
    if not sentences:
        return ""

    terms = _query_terms(query)
    best = ""
    best_score = -10.0

    for sent in sentences:
        if _is_noise_text(sent):
            continue
        canon = _canonical_text(sent)
        hits = sum(1 for t in terms if t in canon)
        score = float(hits) + (1.5 * _tech_density(sent))
        if score > best_score:
            best_score = score
            best = sent

    if not best:
        candidates = [s for s in sentences if not _is_noise_text(s)]
        if not candidates:
            return ""
        best = max(candidates, key=lambda x: _tech_density(x))

    if len(best) <= max_len:
        return best
    return best[:max_len].rstrip() + "..."


def _evidence_relevance_bonus(text: str, query: str) -> float:
    canon = _canonical_text(text)
    hits = sum(1 for t in _query_terms(query) if t in canon)
    return 0.03 * hits + 0.12 * _tech_density(text)


def _best_matches(query: str, vectorizer: TfidfVectorizer, matrix, chunks: List[str], k: int) -> List[Tuple[float, str]]:
    if not chunks:
        return []
    q = clean_text(query)
    if not q:
        return []
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, matrix).ravel()
    if sims.size == 0:
        return []
    kk = max(1, min(k, sims.size))
    idx = np.argpartition(-sims, kk - 1)[:kk]
    idx = idx[np.argsort(-sims[idx])]
    out: List[Tuple[float, str]] = []
    for i in idx:
        score = float(sims[i])
        if score <= 0:
            continue
        out.append((score, chunks[int(i)]))
    return out


def _resume_years_max(text: str) -> Optional[int]:
    t = normalize(text)
    candidates: List[int] = []
    for m in re.finditer(r"\b(\d{1,2})\s*\+?\s*(?:years|yrs|year|года|лет|год)\b", t):
        try:
            candidates.append(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(?:years|yrs|year|года|лет|год)\b", t):
        try:
            candidates.append(max(int(m.group(1)), int(m.group(2))))
        except Exception:
            pass
    if not candidates:
        return None
    return max(candidates)


def _required_years(req: str) -> Optional[int]:
    r = normalize(req)
    m = re.search(r"\b(\d{1,2})\s*\+\s*(?:years|yrs|year|года|лет|год)\b", r)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\b(\d{1,2})\s*(?:years|yrs|year|года|лет|год)\b", r)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _match_experience_requirement(req: str, resume_text: str, profile_strings: List[str]) -> Optional[bool]:
    rl = normalize(req)
    if not any(x in rl for x in ["year", "yrs", "опыт", "лет", "года", "год"]):
        return None
    need = _required_years(req)
    if need is None:
        return None
    resume_max = _resume_years_max(resume_text)
    if resume_max is not None and resume_max >= need:
        return True
    for s in profile_strings:
        if str(need) in normalize(s):
            return True
    return False


def _match_requirement(req: str, skill_strings: List[str], evidence_chunks: List[Dict[str, Any]], resume_text: str) -> bool:
    req_text = clean_text(req)
    if not req_text:
        return True

    exp_match = _match_experience_requirement(req_text, resume_text, skill_strings)
    if exp_match is not None:
        return bool(exp_match)

    req_canon = _canonical_text(req_text)
    req_terms = [x for x in re.split(r"[^a-zA-Zа-яА-Я0-9\+#\.]+", req_canon) if x]

    def ok(target: str) -> bool:
        canon = _canonical_text(target)
        if req_canon and req_canon in canon:
            return True
        if req_terms and all(term in canon for term in req_terms if len(term) > 2):
            return True
        return False

    for s in skill_strings:
        if ok(s):
            return True

    for ev in evidence_chunks or []:
        text = str(ev.get("text") or ev.get("chunk") or "")
        if text and ok(text):
            return True

    return ok(resume_text or "")


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=60000, min_df=1)


def _pick_latest_document(docs: List[Document]) -> Optional[Document]:
    if not docs:
        return None
    docs_sorted = sorted(docs, key=lambda d: (d.parsed_at is not None, d.parsed_at or datetime.datetime.min), reverse=True)
    return docs_sorted[0]


def _job_as_text(job: Job) -> str:
    reqs = _extract_requirements(job.requirements_json or {})
    return clean_text("\n\n".join([f"TITLE: {job.title}", job.description or "", "REQUIREMENTS:", "\n".join(reqs)]))


def _format_evidence(scored_chunks: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ev in enumerate(scored_chunks[:limit]):
        out.append({
            "text": ev.get("text") or "",
            "score": float(ev.get("score") or 0.0),
            "query": ev.get("query") or "",
            "why": ev.get("why") or "",
            "rank": i + 1,
        })
    return out


def _simple_keywords_from_text(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9\+\#\.\-/]{1,30}", text)
    stop = {
        "job", "title", "location", "employment", "type", "company", "description", "responsibilities",
        "requirements", "nice", "have", "salary", "keywords", "remote", "команда", "будет", "работа",
        "важно", "ищем", "candidate", "role", "experience", "skills", "resume"
    }
    out: List[str] = []
    seen = set()
    for token in tokens:
        tl = normalize(token)
        if tl in stop:
            continue
        if len(token) < 2:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(token)
    return out[:80]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items or []:
        value = clean_text(str(item or ""))
        if not value:
            continue
        key = _canonical_text(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _validate_missing_requirements(missing: List[str], requirements: List[str]) -> List[str]:
    canon_to_req: Dict[str, str] = {}
    for req in requirements or []:
        canon = _canonical_text(req)
        if canon and canon not in canon_to_req:
            canon_to_req[canon] = req

    out: List[str] = []
    seen = set()
    for item in missing or []:
        canon = _canonical_text(str(item or ""))
        mapped = canon_to_req.get(canon)
        if not mapped:
            continue
        if canon in seen:
            continue
        seen.add(canon)
        out.append(mapped)
    return out


def _llm_supported_requirements(rationale: str, requirements: List[str]) -> List[str]:
    text = _canonical_text(rationale)
    out: List[str] = []
    for req in requirements:
        canon = _canonical_text(req)
        if canon and canon in text:
            out.append(req)
    return _dedupe_preserve_order(out)


def _compose_final_missing(
    deterministic_missing: List[str],
    llm_missing: List[str],
    rationale: str,
    all_requirements: List[str],
) -> List[str]:
    missing = set(_canonical_text(x) for x in deterministic_missing)
    for item in llm_missing:
        missing.add(_canonical_text(item))

    lower_rationale = (rationale or "").lower()
    if "meets all must-have" in lower_rationale or "meets all required" in lower_rationale:
        for req in all_requirements:
            missing.discard(_canonical_text(req))

    supported = _llm_supported_requirements(rationale, all_requirements)
    if "meets" in lower_rationale:
        for req in supported:
            missing.discard(_canonical_text(req))

    canon_to_req = {_canonical_text(req): req for req in all_requirements}
    out: List[str] = []
    for key in missing:
        req = canon_to_req.get(key)
        if req:
            out.append(req)
    return _dedupe_preserve_order(out)


def _rationale_text(
    must_total: int,
    must_hits: int,
    nice_total: int,
    nice_hits: int,
    other_total: int,
    other_hits: int,
    kw_coverage: float,
    evidence_strength: float,
    final_missing: List[str],
) -> str:
    parts: List[str] = []
    if must_total > 0:
        parts.append(f"Must-have coverage: {must_hits}/{must_total}")
    if nice_total > 0:
        parts.append(f"Nice-to-have coverage: {nice_hits}/{nice_total}")
    if other_total > 0:
        parts.append(f"Other coverage: {other_hits}/{other_total}")
    parts.append(f"Keyword coverage: {kw_coverage:.2f}%")
    parts.append(f"Evidence strength: {round(float(evidence_strength), 3)}")
    if final_missing:
        parts.append("Missing: " + ", ".join(final_missing[:10]))
    else:
        parts.append("No validated missing requirements")
    return ". ".join(parts)


def _apply_hard_rules(
    score: int,
    must_total: int,
    must_hits: int,
    optional_missing_count: int,
    final_missing: List[str],
    rationale: str,
) -> int:
    out = int(score)
    must_cov = (float(must_hits) / float(must_total)) if must_total > 0 else 1.0
    lower = (rationale or "").lower()

    if must_total > 0 and must_hits == 0:
        out = min(out, 20)
    if must_total > 0 and must_cov >= 0.8:
        out = max(out, 50)
    if optional_missing_count <= 2 and must_cov >= 0.6:
        out = max(out, 35)
    if not final_missing:
        out = max(out, 75)
    if "meets most requirements" in lower or "meets core requirements" in lower:
        out = max(out, 50)
    if "lacks only" in lower and out == 0:
        out = 35
    return max(0, min(100, out))


def _filter_and_select_evidence(scored_chunks: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()
    for ev in sorted(scored_chunks, key=lambda x: float(x.get("score") or 0.0), reverse=True):
        text = clean_text(str(ev.get("text") or ""))
        if _is_noise_text(text):
            continue
        key = _canonical_text(text)
        if key in seen:
            continue
        seen.add(key)
        item = dict(ev)
        item["text"] = text
        items.append(item)
        if len(items) >= limit:
            break
    return items


async def index_job(db: AsyncSession, job_id: str) -> int:
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise RuntimeError(f"Job {job_id} not found")
    return len(split_into_chunks(_job_as_text(job)))


async def index_resume(db: AsyncSession, document_id: str) -> int:
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise RuntimeError(f"Document {document_id} not found")
    if not doc.raw_text:
        raise RuntimeError("document raw_text not ready")
    return len(split_into_chunks(clean_text(doc.raw_text)))


async def retrieve_evidence(db: AsyncSession, job_id: str, candidate_id: str, k: int = 5) -> List[Dict[str, Any]]:
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise RuntimeError(f"Job {job_id} not found")

    docs_result = await db.execute(select(Document).where(Document.candidate_id == candidate_id).where(Document.raw_text.isnot(None)))
    docs = list(docs_result.scalars().all())
    doc = _pick_latest_document(docs)
    if not doc or not doc.raw_text:
        return []

    resume_text = clean_text(doc.raw_text)
    resume_chunks = split_into_chunks(resume_text)
    if not resume_chunks:
        return []

    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(resume_chunks)

    req_groups = _extract_requirement_groups(job.requirements_json or {})
    queries: List[str] = []
    if job.title:
        queries.append(clean_text(job.title))
    if job.description:
        queries.append(clean_text(job.description[:1000]))
    queries.extend(req_groups.get("must_have", []))
    queries.extend(req_groups.get("nice_to_have", []))
    queries.extend(req_groups.get("other", []))

    uniq_queries: List[str] = []
    seen = set()
    for q in queries:
        value = clean_text(q)
        if not value:
            continue
        key = _canonical_text(value)
        if key in seen:
            continue
        seen.add(key)
        uniq_queries.append(value)

    scored: Dict[str, Dict[str, Any]] = {}
    for query in uniq_queries:
        matches = _best_matches(query, vectorizer, matrix, resume_chunks, k=max(3, min(12, k * 3)))
        for sim, chunk in matches:
            snippet = _best_display_snippet(chunk, query)
            if not snippet:
                continue
            if _is_noise_text(snippet):
                continue
            final_score = float(sim) + _evidence_relevance_bonus(snippet, query)
            key = _canonical_text(snippet)
            current = scored.get(key)
            if current is None or final_score > float(current["score"]):
                scored[key] = {
                    "text": snippet,
                    "score": final_score,
                    "query": query,
                    "why": f"matched query: {query}",
                }

    return _filter_and_select_evidence(list(scored.values()), limit=max(1, k))


async def calculate_score(
    requirements_json: Dict[str, Any],
    profile_json: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
    job_title: str = "",
    job_description: str = "",
    resume_text: str = "",
) -> Tuple[int, str, List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    req_groups = _extract_requirement_groups(requirements_json or {})
    must_reqs = req_groups.get("must_have", [])
    nice_reqs = req_groups.get("nice_to_have", [])
    other_reqs = req_groups.get("other", [])
    all_reqs = req_groups.get("all", [])
    skill_strings = _profile_skill_strings(profile_json or {})
    resume_joined = clean_text(resume_text or " ".join([str(e.get("text") or "") for e in evidence_chunks or []]))

    must_hits = [r for r in must_reqs if _match_requirement(r, skill_strings, evidence_chunks, resume_joined)]
    nice_hits = [r for r in nice_reqs if _match_requirement(r, skill_strings, evidence_chunks, resume_joined)]
    other_hits = [r for r in other_reqs if _match_requirement(r, skill_strings, evidence_chunks, resume_joined)]

    must_missing = [r for r in must_reqs if r not in must_hits]
    nice_missing = [r for r in nice_reqs if r not in nice_hits]
    other_missing = [r for r in other_reqs if r not in other_hits]

    must_cov = (len(must_hits) / len(must_reqs)) if must_reqs else 1.0
    nice_cov = (len(nice_hits) / len(nice_reqs)) if nice_reqs else 0.0
    other_cov = (len(other_hits) / len(other_reqs)) if other_reqs else 0.0

    evidence_text = "\n".join([str(e.get("text") or "") for e in evidence_chunks or []])
    grounded_keywords = _simple_keywords_from_text("\n".join([job_title, job_description, "\n".join(all_reqs)]))
    kw_cov = keyword_coverage(resume_joined, grounded_keywords)
    ev_scores = [float(e.get("score") or 0.0) for e in evidence_chunks or []]
    evidence_strength = float(np.mean(ev_scores[:5])) if ev_scores else 0.0

    deterministic_f = 100.0 * (0.72 * must_cov + 0.18 * nice_cov + 0.05 * other_cov + 0.05 * min(1.0, kw_cov / 100.0))
    deterministic_f += min(5.0, evidence_strength * 10.0)
    deterministic_score = int(round(max(0.0, min(100.0, deterministic_f))))

    deterministic_missing = _dedupe_preserve_order(must_missing + nice_missing + other_missing)
    scoring_mode = "FALLBACK"
    llm_adjustment = 0
    llm_missing: List[str] = []
    llm_rationale = ""
    selected_evidence = _filter_and_select_evidence(evidence_chunks or [], limit=5)

    if selected_evidence:
        try:
            llm = await asyncio.to_thread(
                llm_score_application,
                job_title,
                clean_text(job_description),
                req_groups,
                selected_evidence,
            )
        except Exception:
            llm = None
        if llm is not None:
            scoring_mode = "LLM"
            llm_adjustment = max(-10, min(10, int(llm.get("score_adjustment") or 0)))
            llm_missing = _validate_missing_requirements(list(llm.get("missing_requirements") or []), all_reqs)
            llm_rationale = clean_text(str(llm.get("rationale") or ""))
            indices = llm.get("evidence_indices") or []
            if indices:
                chosen: List[Dict[str, Any]] = []
                for idx in indices:
                    if 1 <= idx <= len(selected_evidence):
                        chosen.append(selected_evidence[idx - 1])
                if chosen:
                    selected_evidence = _filter_and_select_evidence(chosen, limit=5)

    final_missing = _compose_final_missing(deterministic_missing, llm_missing, llm_rationale, all_reqs)
    rationale_body = llm_rationale or _rationale_text(
        must_total=len(must_reqs),
        must_hits=len(must_hits),
        nice_total=len(nice_reqs),
        nice_hits=len(nice_hits),
        other_total=len(other_reqs),
        other_hits=len(other_hits),
        kw_coverage=kw_cov,
        evidence_strength=evidence_strength,
        final_missing=final_missing,
    )

    final_score = deterministic_score + llm_adjustment
    final_score = _apply_hard_rules(
        score=final_score,
        must_total=len(must_reqs),
        must_hits=len(must_hits),
        optional_missing_count=len(nice_missing) + len(other_missing),
        final_missing=final_missing,
        rationale=rationale_body,
    )

    rationale_prefix = "[LLM] " if scoring_mode == "LLM" else "[FALLBACK] "
    evidence_out = _format_evidence(selected_evidence, limit=12)

    debug = {
        "deterministic_score": deterministic_score,
        "llm_adjustment": llm_adjustment,
        "must_have_total": len(must_reqs),
        "must_have_hits": len(must_hits),
        "nice_to_have_total": len(nice_reqs),
        "nice_to_have_hits": len(nice_hits),
        "other_total": len(other_reqs),
        "other_hits": len(other_hits),
        "keyword_coverage_percent": kw_cov,
        "deterministic_missing": deterministic_missing,
        "llm_missing_raw": llm_missing,
        "final_missing": final_missing,
    }

    return final_score, rationale_prefix + rationale_body, final_missing, evidence_out, scoring_mode, debug


async def save_scoring(
    db: AsyncSession,
    application_id: str,
    score: int,
    rationale: str,
    missing: List[str],
    evidence: List[Dict[str, Any]],
    debug: Dict[str, Any],
) -> None:
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app = app_result.scalar_one()
    app.score = score
    app.score_rationale = rationale
    app.missing_requirements_json = missing
    payload = {
        "evidence": evidence,
        "debug": debug,
    }
    app.evidence_snippets_json = payload
    app.status = "SCORING_DONE"
    app.updated_at = _utcnow()

    rag_run = RagRun(
        id=str(uuid.uuid4()),
        application_id=application_id,
        top_k_chunks_json=payload,
        prompt_version="ollama_rag_ported_v1",
        model_version="ollama_local",
        created_at=_utcnow(),
    )
    db.add(rag_run)


async def score_application(db: AsyncSession, application_id: str) -> Dict[str, Any]:
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app = app_result.scalar_one_or_none()
    if not app:
        raise RuntimeError(f"Application {application_id} not found")

    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise RuntimeError(f"Job {app.job_id} not found")

    profile_result = await db.execute(select(Profile).where(Profile.candidate_id == app.candidate_id))
    profile = profile_result.scalar_one_or_none()
    if not profile:
        raise RuntimeError("candidate profile not ready")

    docs_result = await db.execute(select(Document).where(Document.candidate_id == app.candidate_id).where(Document.raw_text.isnot(None)))
    docs = list(docs_result.scalars().all())
    doc = _pick_latest_document(docs)
    if not doc or not doc.raw_text:
        raise RuntimeError("candidate documents not found")

    resume_text = clean_text(doc.raw_text)
    evidence_chunks = await retrieve_evidence(db, job_id=app.job_id, candidate_id=app.candidate_id, k=5)

    score, rationale, missing, evidence, scoring_mode, debug = await calculate_score(
        requirements_json=job.requirements_json or {},
        profile_json=profile.profile_json,
        evidence_chunks=evidence_chunks,
        job_title=job.title or "",
        job_description=job.description or "",
        resume_text=resume_text,
    )

    await save_scoring(
        db=db,
        application_id=application_id,
        score=score,
        rationale=rationale,
        missing=missing,
        evidence=evidence,
        debug=debug,
    )

    await db.commit()

    return {
        "application_id": application_id,
        "score": int(score),
        "rationale": rationale,
        "missing_requirements": missing,
        "evidence_snippets": evidence,
        "scoring_mode": scoring_mode,
        **debug,
    }
