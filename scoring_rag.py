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
from rag_text_utils import clean_text, extract_resume_query, keyword_coverage, normalize, split_into_chunks


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


def _extract_requirements(requirements_json: Dict[str, Any]) -> List[str]:
    rj = requirements_json or {}

    reqs: List[str] = []
    for k in ["must_have", "must_haves", "nice_to_have", "nice_to_haves", "experience", "languages", "requirements"]:
        if k in rj:
            reqs.extend(_extract_strings_only(rj.get(k), max_items=200))

    if not reqs:
        reqs = _extract_strings_only(rj, max_items=300)

    cleaned: List[str] = []
    seen = set()
    for r in reqs:
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

    return cleaned[:60]


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


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=60000,
        min_df=1,
    )


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
            a = int(m.group(1))
            b = int(m.group(2))
            candidates.append(max(a, b))
        except Exception:
            pass

    if not candidates:
        return None
    return max(candidates)


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
    if "year" not in rl and "yrs" not in rl and "опыт" not in rl and "лет" not in rl and "года" not in rl and "год" not in rl:
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

    rl = r.lower()

    for s in skill_strings:
        if rl in s.lower():
            return True

    for ev in evidence_chunks or []:
        t = ev.get("text") or ev.get("chunk") or ""
        if isinstance(t, str) and rl in t.lower():
            return True

    if rl in (resume_text or "").lower():
        return True

    return False


def _pick_latest_document(docs: List[Document]) -> Optional[Document]:
    if not docs:
        return None
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            d.parsed_at is not None,
            d.parsed_at or datetime.datetime.min,
        ),
        reverse=True,
    )
    return docs_sorted[0]


def _job_as_text(job: Job) -> str:
    reqs = _extract_requirements(job.requirements_json or {})
    return clean_text(
        "\n\n".join(
            [
                f"TITLE: {job.title}",
                job.description or "",
                "REQUIREMENTS:",
                "\n".join(reqs),
            ]
        )
    )


def _format_evidence(scored_chunks: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ev in enumerate(scored_chunks[:limit]):
        out.append(
            {
                "text": ev.get("text") or "",
                "score": float(ev.get("score") or 0.0),
                "query": ev.get("query") or "",
                "rank": i + 1,
            }
        )
    return out


def _rationale_text(req_total: int, hit_count: int, evidence_strength: float, hits: List[str], missing: List[str]) -> str:
    parts: List[str] = []
    if req_total > 0:
        parts.append(f"Coverage: {int(round((hit_count / req_total) * 100))}% ({hit_count}/{req_total})")
    parts.append(f"Evidence strength: {round(float(evidence_strength), 3)}")
    if hits:
        parts.append("Matched: " + ", ".join(hits[:8]))
    if missing:
        parts.append("Missing: " + ", ".join(missing[:10]))
    return ". ".join([p for p in parts if p]).strip() or "No sufficient data to score"


async def index_job(db: AsyncSession, job_id: str) -> int:
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise RuntimeError(f"Job {job_id} not found")
    text = _job_as_text(job)
    chunks = split_into_chunks(text)
    return len(chunks)


async def index_resume(db: AsyncSession, document_id: str) -> int:
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise RuntimeError(f"Document {document_id} not found")
    if not doc.raw_text:
        raise RuntimeError("document raw_text not ready")
    chunks = split_into_chunks(clean_text(doc.raw_text))
    return len(chunks)


async def retrieve_evidence(db: AsyncSession, job_id: str, candidate_id: str, k: int = 5) -> List[Dict[str, Any]]:
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise RuntimeError(f"Job {job_id} not found")

    docs_result = await db.execute(
        select(Document)
        .where(Document.candidate_id == candidate_id)
        .where(Document.raw_text.isnot(None))
    )
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

    reqs = _extract_requirements(job.requirements_json or {})
    queries: List[str] = []
    if job.title:
        queries.append(clean_text(job.title))
    if job.description:
        queries.append(clean_text(job.description[:1200]))
    queries.extend(reqs)

    resume_query = extract_resume_query(resume_text)
    if resume_query:
        queries.append(resume_query)

    uniq_queries: List[str] = []
    seen = set()
    for q in queries:
        qq = clean_text(q)
        if not qq:
            continue
        ql = qq.lower()
        if ql in seen:
            continue
        seen.add(ql)
        uniq_queries.append(qq)

    scored: Dict[str, Dict[str, Any]] = {}
    per_query_k = max(3, min(12, k * 3))

    for q in uniq_queries:
        matches = _best_matches(q, vectorizer, matrix, resume_chunks, k=per_query_k)
        for s, ch in matches:
            key = ch
            cur = scored.get(key)
            if cur is None or float(s) > float(cur["score"]):
                scored[key] = {
                    "text": ch,
                    "score": float(s),
                    "query": q,
                }

    out = sorted(scored.values(), key=lambda x: float(x["score"]), reverse=True)
    return out[: max(1, k)]


async def calculate_score(
    requirements_json: Dict[str, Any],
    profile_json: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
    job_title: str = "",
    job_description: str = "",
    resume_text: str = "",
) -> Tuple[int, str, List[str], List[Dict[str, Any]]]:
    reqs = _extract_requirements(requirements_json or {})
    skill_strings = _profile_skill_strings(profile_json or {})

    use_ollama = True

    if use_ollama and evidence_chunks:
        try:
            llm = await asyncio.to_thread(
                llm_score_application,
                clean_text(resume_text),
                job_title,
                clean_text(job_description),
                reqs,
                evidence_chunks,
            )
        except Exception:
            llm = None

        if llm is not None:
            score = int(llm["score"])
            rationale = str(llm["rationale"])
            missing = list(llm["missing_requirements"])
            evidence = _format_evidence(evidence_chunks or [], limit=12)
            return score, rationale, missing, evidence

    resume_joined = clean_text(
        resume_text or " ".join([(e.get("text") or "") for e in (evidence_chunks or []) if isinstance(e.get("text"), str)])
    )

    hits: List[str] = []
    missing: List[str] = []

    if reqs:
        for r in reqs:
            if _match_requirement(r, skill_strings, evidence_chunks, resume_joined):
                hits.append(r)
            else:
                missing.append(r)
        coverage = len(hits) / len(reqs)
    else:
        coverage = 0.0

    ev_scores = [float(e.get("score") or 0.0) for e in (evidence_chunks or [])]
    evidence_strength = float(np.mean(ev_scores)) if ev_scores else 0.0

    score_f = 100.0 * (0.7 * float(coverage) + 0.3 * min(1.0, float(evidence_strength)))
    score_f = max(0.0, min(100.0, score_f))
    score = int(round(score_f))

    rationale = _rationale_text(len(reqs), len(hits), evidence_strength, hits, missing)
    evidence = _format_evidence(evidence_chunks or [], limit=12)

    return score, rationale, missing, evidence


async def save_scoring(
    db: AsyncSession,
    application_id: str,
    score: int,
    rationale: str,
    missing: List[str],
    evidence: List[Dict[str, Any]],
) -> None:
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
        prompt_version="ollama_rag_v1",
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

    docs_result = await db.execute(
        select(Document)
        .where(Document.candidate_id == app.candidate_id)
        .where(Document.raw_text.isnot(None))
    )
    docs = list(docs_result.scalars().all())
    doc = _pick_latest_document(docs)
    if not doc or not doc.raw_text:
        raise RuntimeError("candidate documents not found")

    resume_text = clean_text(doc.raw_text)
    evidence_chunks = await retrieve_evidence(
        db,
        job_id=app.job_id,
        candidate_id=app.candidate_id,
        k=5,
    )

    score, rationale, missing, evidence = await calculate_score(
        job.requirements_json or {},
        profile.profile_json,
        evidence_chunks,
        job_title=job.title or "",
        job_description=job.description or "",
        resume_text=resume_text,
    )

    await save_scoring(
        db,
        application_id,
        score,
        rationale,
        missing,
        evidence,
    )

    await db.commit()

    return {
        "application_id": application_id,
        "score": int(score),
        "rationale": rationale,
        "missing_requirements": missing,
        "evidence_snippets": evidence,
    }