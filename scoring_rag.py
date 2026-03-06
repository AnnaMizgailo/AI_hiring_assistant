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
from ollama_scoring import llm_score_application, simple_keywords_from_text
from rag_text_utils import clean_text, keyword_coverage, normalize, split_into_chunks


_CONTAINER_KEYS = {
    "must_have", "must_haves", "nice_to_have", "nice_to_haves", "requirements", "requirement",
    "responsibilities", "responsibility", "languages", "language", "experience", "education",
    "skills", "tech_stack", "stack", "tools", "summary",
}

_BAD_SNIPPET_PATTERNS = [
    r"\bemail\b", r"\bтелефон\b", r"\bгород\b", r"\bанглийский\b", r"\bдополнительно\b",
    r"\bjira\/youtrack\b", r"\bпланирован", r"\bретро\b", r"\bревью\b", r"\bожидания\b",
]

_ROLE_ALIAS = {
    "product management": ["product manager", "product owner", "backlog", "prioritization", "stakeholder"],
    "analytics": ["metrics", "dashboard", "insights", "analysis", "analyst"],
    "roadmap": ["roadmap", "strategy", "prioritization", "planning"],
    "jtbd": ["jtbd", "jobs to be done", "job to be done", "discovery", "customer research"],
    "test design": ["test design", "test cases", "test scenario", "test plan"],
    "ci/cd": ["ci", "cd", "pipeline", "github actions", "gitlab ci", "jenkins"],
    "code review": ["code review", "review", "pull request", "pr"],
    "analytics / experiments": ["a/b testing", "ab testing", "experiment", "hypothesis"],
}


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
    cleaned: List[str] = []
    seen = set()
    for r in items:
        rr = clean_text(r)
        if not rr:
            continue
        rl = rr.lower()
        if rl in _CONTAINER_KEYS:
            continue
        if len(rr) < 2:
            continue
        if rl in seen:
            continue
        seen.add(rl)
        cleaned.append(rr)
    return cleaned[:limit]


def _extract_requirement_groups(requirements_json: Dict[str, Any]) -> Dict[str, List[str]]:
    rj = requirements_json or {}
    must = _clean_requirement_list(_extract_strings_only(rj.get("must_have") or rj.get("must_haves") or [], max_items=200))
    nice = _clean_requirement_list(_extract_strings_only(rj.get("nice_to_have") or rj.get("nice_to_haves") or [], max_items=200))
    other = _clean_requirement_list(
        _extract_strings_only(rj.get("requirements") or [], max_items=200) +
        _extract_strings_only(rj.get("experience") or [], max_items=100) +
        _extract_strings_only(rj.get("languages") or [], max_items=100)
    )
    if not must and not nice and not other:
        must = _clean_requirement_list(_extract_strings_only(rj, max_items=300))
    all_items = _clean_requirement_list(must + nice + other, limit=120)
    return {"must_have": must, "nice_to_have": nice, "other": other, "all": all_items}


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
        sl = s.lower()
        if sl in seen:
            continue
        seen.add(sl)
        uniq.append(s)
    return uniq


def _canonical_text(text: str) -> str:
    t = normalize(text)
    replacements = [
        ("typescript", "ts"), ("type script", "ts"),
        ("next.js", "nextjs"), ("next js", "nextjs"),
        ("postgresql", "postgres"), ("postgre sql", "postgres"),
        ("ci/cd", "cicd"), ("ci cd", "cicd"),
        ("a/b testing", "abtesting"), ("ab testing", "abtesting"), ("a b testing", "abtesting"),
        ("rest api", "rest"), ("product management", "productmanager"), ("product manager", "productmanager"),
        ("jobs to be done", "jtbd"), ("job to be done", "jtbd"), ("test design", "testdesign"),
    ]
    for src, dst in replacements:
        t = t.replace(src, dst)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _req_aliases(req: str) -> List[str]:
    canon = _canonical_text(req)
    out = [canon]
    for key, aliases in _ROLE_ALIAS.items():
        if canon == _canonical_text(key):
            out.extend([_canonical_text(x) for x in aliases])
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


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
        for canon in _req_aliases(req):
            canon_to_req.setdefault(canon, req)
    out: List[str] = []
    seen = set()
    for item in missing or []:
        mapped = canon_to_req.get(_canonical_text(str(item or "")))
        if not mapped:
            continue
        key = _canonical_text(mapped)
        if key in seen:
            continue
        seen.add(key)
        out.append(mapped)
    return out


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=60000, min_df=1)


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
        s = float(sims[i])
        if s <= 0:
            continue
        out.append((s, chunks[int(i)]))
    return out


def _resume_years_max(text: str) -> Optional[int]:
    t = (text or "").lower()
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
    return max(candidates) if candidates else None


def _required_years(req: str) -> Optional[int]:
    r = (req or "").lower()
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
    rl = (req or "").lower()
    if not any(x in rl for x in ["year", "yrs", "опыт", "лет", "года", "год"]):
        return None
    need = _required_years(req)
    if need is None:
        return None
    resume_max = _resume_years_max(resume_text)
    if resume_max is not None and resume_max >= need:
        return True
    for s in profile_strings:
        if str(need) in s:
            return True
    return False


def _match_requirement(req: str, skill_strings: List[str], evidence_chunks: List[Dict[str, Any]], resume_text: str) -> bool:
    r = clean_text(req)
    if not r:
        return True
    exp_match = _match_experience_requirement(r, resume_text, skill_strings)
    if exp_match is not None:
        return bool(exp_match)
    aliases = _req_aliases(r)
    texts = [resume_text or ""] + [str(x) for x in skill_strings] + [str(ev.get("text") or ev.get("chunk") or "") for ev in evidence_chunks or []]
    canon_texts = [_canonical_text(t) for t in texts if t]
    for alias in aliases:
        alias_tokens = [x for x in re.split(r"[^a-zA-Zа-яА-Я0-9+.#]+", alias) if x]
        for ct in canon_texts:
            if alias and alias in ct:
                return True
            if alias_tokens and all(tok in ct for tok in alias_tokens if len(tok) > 2):
                return True
    return False


def _pick_latest_document(docs: List[Document]) -> Optional[Document]:
    if not docs:
        return None
    docs_sorted = sorted(docs, key=lambda d: (d.parsed_at is not None, d.parsed_at or datetime.datetime.min), reverse=True)
    return docs_sorted[0]


def _job_as_text(job: Job) -> str:
    reqs = _extract_requirements(job.requirements_json or {})
    return clean_text("\n\n".join([f"TITLE: {job.title}", job.description or "", "REQUIREMENTS:", "\n".join(reqs)]))


def _is_bad_snippet(text: str) -> bool:
    t = clean_text(text)
    if len(t) < 18:
        return True
    if len(t.split()) <= 2:
        return True
    if re.fullmatch(r"[ABC][12]?", t):
        return True
    tl = _canonical_text(t)
    for pat in _BAD_SNIPPET_PATTERNS:
        if re.search(pat, tl):
            return True
    return False


def _best_display_snippet(chunk_text: str, query: str, max_len: int = 280) -> str:
    text = clean_text(chunk_text)
    if not text:
        return ""
    parts = [clean_text(x) for x in re.split(r"(?<=[\.!?])\s+|\n+", text) if clean_text(x)]
    if not parts:
        return text[:max_len]
    terms = simple_keywords_from_text(query)
    best = ""
    best_score = -1.0
    for sent in parts:
        st = _canonical_text(sent)
        hits = sum(1 for term in terms if _canonical_text(term) in st)
        tech_hits = len(simple_keywords_from_text(sent))
        score = float(hits) + min(2.0, tech_hits * 0.15)
        if _is_bad_snippet(sent):
            score -= 1.0
        if score > best_score:
            best_score = score
            best = sent
    best = clean_text(best or parts[0])
    if len(best) <= max_len:
        return best
    return best[:max_len].rstrip() + "..."


def _evidence_relevance_bonus(text: str, query: str) -> float:
    ct = _canonical_text(text)
    terms = [_canonical_text(x) for x in simple_keywords_from_text(query)]
    hits = sum(1 for t in terms if t and t in ct)
    bonus = 0.03 * hits + min(0.1, len(simple_keywords_from_text(text)) * 0.01)
    if _is_bad_snippet(text):
        bonus -= 0.08
    return bonus


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


def _bound_adjustment(deterministic_score: int, raw_adjustment: int) -> int:
    if deterministic_score < 30:
        lim = 3
    elif deterministic_score <= 60:
        lim = 5
    else:
        lim = 7
    return max(-lim, min(lim, int(raw_adjustment)))


def _apply_hard_rules(score: int, must_total: int, must_hits: int, final_missing: List[str], rationale: str) -> int:
    out = int(score)
    must_cov = float(must_hits) / float(must_total) if must_total > 0 else 1.0
    rl = (rationale or "").lower()
    if must_total > 0 and must_hits == 0:
        out = min(out, 20)
    if must_total > 0 and must_cov >= 0.8:
        out = max(out, 50)
    if not final_missing and must_total > 0:
        out = max(out, 75)
    if "meets most requirements" in rl or "meets core requirements" in rl:
        out = max(out, 50)
    if "lacks only" in rl and out < 35:
        out = 35
    return max(0, min(100, out))


def _merge_missing(must_missing: List[str], optional_missing: List[str], llm_missing: List[str], req_groups: Dict[str, List[str]]) -> List[str]:
    must_set = {_canonical_text(x) for x in req_groups.get("must_have", [])}
    nice_set = {_canonical_text(x) for x in req_groups.get("nice_to_have", []) + req_groups.get("other", [])}
    out = _dedupe_preserve_order(must_missing + optional_missing)
    out_keys = {_canonical_text(x) for x in out}
    for item in llm_missing:
        key = _canonical_text(item)
        if key in must_set:
            continue
        if key in nice_set and key not in out_keys:
            out.append(item)
            out_keys.add(key)
    return out


def _rationale_text(must_total: int, must_hits: int, nice_total: int, nice_hits: int, keyword_cov: float, evidence_strength: float, final_missing: List[str]) -> str:
    parts = []
    if must_total > 0:
        parts.append(f"Must-have coverage: {must_hits}/{must_total}")
    if nice_total > 0:
        parts.append(f"Nice-to-have coverage: {nice_hits}/{nice_total}")
    parts.append(f"Keyword coverage: {round(keyword_cov, 2)}%")
    parts.append(f"Evidence strength: {round(float(evidence_strength), 3)}")
    if final_missing:
        parts.append("Missing: " + ", ".join(final_missing[:10]))
    return ". ".join(parts)


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
    queries.extend(req_groups.get("must_have", []))
    queries.extend(req_groups.get("nice_to_have", []))
    if not queries and job.description:
        queries.append(clean_text(job.description[:240]))

    uniq_queries: List[str] = []
    seen = set()
    for q in queries:
        qq = clean_text(q)
        if not qq:
            continue
        qk = _canonical_text(qq)
        if qk in seen:
            continue
        seen.add(qk)
        uniq_queries.append(qq)

    scored: Dict[str, Dict[str, Any]] = {}
    per_query_k = max(3, min(10, k * 3))
    for q in uniq_queries:
        matches = _best_matches(q, vectorizer, matrix, resume_chunks, k=per_query_k)
        for s, ch in matches:
            display_text = _best_display_snippet(ch, q)
            if not display_text or _is_bad_snippet(display_text):
                continue
            final_score = float(s) + _evidence_relevance_bonus(display_text, q)
            if final_score < 0.08:
                continue
            key = _canonical_text(display_text)
            cur = scored.get(key)
            if cur is None or final_score > float(cur["score"]):
                scored[key] = {
                    "text": display_text,
                    "score": float(final_score),
                    "query": q,
                    "why": f"matched query: {q}",
                }
    out = sorted(scored.values(), key=lambda x: float(x["score"]), reverse=True)
    return out[:max(1, k)]


async def calculate_score(
    requirements_json: Dict[str, Any],
    profile_json: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
    job_title: str = "",
    job_description: str = "",
    resume_text: str = "",
) -> Tuple[int, str, List[str], List[Dict[str, Any]], str, int, int, int, int, int, int, float, List[str], List[str], List[str]]:
    req_groups = _extract_requirement_groups(requirements_json or {})
    must_reqs = req_groups.get("must_have", [])
    nice_reqs = req_groups.get("nice_to_have", [])
    other_reqs = req_groups.get("other", [])
    all_reqs = req_groups.get("all", [])
    skill_strings = _profile_skill_strings(profile_json or {})
    resume_joined = clean_text(resume_text or " ".join([str(e.get("text") or "") for e in evidence_chunks or []]))

    must_hits, must_missing = [], []
    nice_hits, nice_missing = [], []
    other_hits, other_missing = [], []

    for r in must_reqs:
        (must_hits if _match_requirement(r, skill_strings, evidence_chunks, resume_joined) else must_missing).append(r)
    for r in nice_reqs:
        (nice_hits if _match_requirement(r, skill_strings, evidence_chunks, resume_joined) else nice_missing).append(r)
    for r in other_reqs:
        (other_hits if _match_requirement(r, skill_strings, evidence_chunks, resume_joined) else other_missing).append(r)

    must_cov = float(len(must_hits)) / float(len(must_reqs)) if must_reqs else 1.0
    nice_cov = float(len(nice_hits)) / float(len(nice_reqs)) if nice_reqs else 0.0
    other_cov = float(len(other_hits)) / float(len(other_reqs)) if other_reqs else 0.0

    evidence_text = "\n".join([str(x.get("text") or "") for x in evidence_chunks or []])
    grounded_keywords = simple_keywords_from_text(evidence_text) or all_reqs
    kw_cov = keyword_coverage(resume_joined, grounded_keywords)

    ev_scores = [float(e.get("score") or 0.0) for e in (evidence_chunks or [])]
    evidence_strength = float(np.mean(ev_scores[:5])) if ev_scores else 0.0

    base_score = 100.0 * (0.7 * must_cov + 0.15 * nice_cov + 0.05 * other_cov + 0.10 * (kw_cov / 100.0))
    base_score += min(5.0, evidence_strength * 10.0)
    deterministic_score = int(round(max(0.0, min(100.0, base_score))))

    deterministic_missing = _dedupe_preserve_order(must_missing + nice_missing + other_missing)
    final_missing = list(deterministic_missing)
    evidence_out = _format_evidence(evidence_chunks or [], limit=12)
    rationale_body = _rationale_text(len(must_reqs), len(must_hits), len(nice_reqs), len(nice_hits), kw_cov, evidence_strength, final_missing)
    llm_adjustment = 0
    llm_missing_raw: List[str] = []
    scoring_mode = "FALLBACK"

    if evidence_chunks:
        try:
            llm = await asyncio.to_thread(
                llm_score_application,
                resume_joined,
                job_title,
                req_groups,
                evidence_chunks,
                kw_cov,
            )
        except Exception:
            llm = None
        if llm is not None:
            scoring_mode = "LLM"
            rationale_body = clean_text(str(llm.get("rationale") or rationale_body))
            llm_missing_raw = _validate_missing_requirements(list(llm.get("missing_requirements") or []), all_reqs)
            llm_adjustment = _bound_adjustment(deterministic_score, int(llm.get("score_adjustment") or 0))
            final_missing = _merge_missing(must_missing, nice_missing + other_missing, llm_missing_raw, req_groups)
            llm_evidence = []
            for i, item in enumerate(list(llm.get("evidence_snippets") or [])[:12], start=1):
                if not isinstance(item, dict):
                    continue
                txt = clean_text(str(item.get("text") or ""))
                why = clean_text(str(item.get("why") or ""))
                if not txt or _is_bad_snippet(txt):
                    continue
                llm_evidence.append({"text": txt, "score": 0.0, "query": "", "why": why, "rank": i})
            if llm_evidence:
                evidence_out = llm_evidence

    final_score = _apply_hard_rules(deterministic_score + llm_adjustment, len(must_reqs), len(must_hits), final_missing, rationale_body)
    rationale = f"[{scoring_mode}] {rationale_body}"

    return (
        final_score,
        rationale,
        final_missing,
        evidence_out,
        scoring_mode,
        deterministic_score,
        llm_adjustment,
        len(must_reqs),
        len(must_hits),
        len(nice_reqs),
        len(nice_hits),
        kw_cov,
        deterministic_missing,
        llm_missing_raw,
        final_missing,
    )


async def save_scoring(db: AsyncSession, application_id: str, score: int, rationale: str, missing: List[str], evidence: List[Dict[str, Any]]) -> None:
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app = app_result.scalar_one()
    app.score = score
    app.score_rationale = rationale
    app.missing_requirements_json = missing
    app.evidence_snippets_json = evidence
    app.status = "SCORING_DONE"
    app.updated_at = _utcnow()
    rag_run = RagRun(
        id=str(uuid.uuid4()),
        application_id=application_id,
        top_k_chunks_json=evidence,
        prompt_version="ollama_rag_v3",
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
    evidence_chunks = await retrieve_evidence(db, app.job_id, app.candidate_id, k=5)

    (
        score,
        rationale,
        missing,
        evidence,
        scoring_mode,
        deterministic_score,
        llm_adjustment,
        must_have_total,
        must_have_hits,
        nice_to_have_total,
        nice_to_have_hits,
        keyword_coverage_percent,
        deterministic_missing,
        llm_missing_raw,
        final_missing,
    ) = await calculate_score(
        requirements_json=job.requirements_json or {},
        profile_json=profile.profile_json,
        evidence_chunks=evidence_chunks,
        job_title=job.title or "",
        job_description=job.description or "",
        resume_text=resume_text,
    )

    await save_scoring(db, application_id, score, rationale, missing, evidence)
    await db.commit()

    return {
        "application_id": application_id,
        "score": int(score),
        "rationale": rationale,
        "missing_requirements": missing,
        "evidence_snippets": evidence,
        "scoring_mode": scoring_mode,
        "deterministic_score": deterministic_score,
        "llm_adjustment": llm_adjustment,
        "must_have_total": must_have_total,
        "must_have_hits": must_have_hits,
        "nice_to_have_total": nice_to_have_total,
        "nice_to_have_hits": nice_to_have_hits,
        "other_total": 0,
        "other_hits": 0,
        "keyword_coverage_percent": keyword_coverage_percent,
        "deterministic_missing": deterministic_missing,
        "llm_missing_raw": llm_missing_raw,
        "final_missing": final_missing,
    }
