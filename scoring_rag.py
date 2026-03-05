from typing import Any, Dict, List, Tuple, Optional
import uuid
import datetime
import re

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database import Application, Job, Profile, RagRun, Document


_CONTAINER_KEYS = {
    "must_have",
    "nice_to_have",
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
    "nice_to_haves",
    "must_haves",
}


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _split_into_chunks(text: str, max_chars: int = 900, overlap: int = 160) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: List[str] = []
    buf = ""

    for p in parts:
        p = _normalize_ws(p)
        if not p:
            continue
        if not buf:
            buf = p
            continue
        if len(buf) + 1 + len(p) <= max_chars:
            buf = buf + " " + p
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 and len(buf) > overlap else ""
            buf = (tail + " " + p).strip()

    if buf:
        chunks.append(buf)

    out: List[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        if len(ch) <= max_chars:
            out.append(ch)
            continue
        start = 0
        while start < len(ch):
            end = min(len(ch), start + max_chars)
            piece = ch[start:end].strip()
            if piece:
                out.append(piece)
            if end >= len(ch):
                break
            start = max(0, end - overlap)

    return out


def _extract_strings_only(obj: Any, max_items: int = 400) -> List[str]:
    out: List[str] = []

    def rec(x: Any) -> None:
        nonlocal out
        if len(out) >= max_items:
            return
        if x is None:
            return
        if isinstance(x, str):
            s = _normalize_ws(x)
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
            return

    rec(obj)

    seen = set()
    uniq: List[str] = []
    for s in out:
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
    q = _normalize_ws(query)
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


def _profile_skill_strings(profile_json: Dict[str, Any]) -> List[str]:
    profile_json = profile_json or {}
    items: List[str] = []
    for key in ["skills", "technologies", "stack", "tools", "languages", "frameworks", "certifications", "summary", "projects", "experience"]:
        if key in profile_json:
            items.extend(_extract_strings_only(profile_json.get(key), max_items=250))
    if not items:
        items = _extract_strings_only(profile_json, max_items=500)

    seen = set()
    uniq: List[str] = []
    for s in items:
        sl = s.lower()
        if sl in seen:
            continue
        seen.add(sl)
        uniq.append(s)
    return uniq


def _extract_requirements(requirements_json: Dict[str, Any]) -> List[str]:
    rj = requirements_json or {}

    reqs: List[str] = []
    for k in ["must_have", "must_haves", "nice_to_have", "nice_to_haves", "experience", "languages"]:
        if k in rj:
            reqs.extend(_extract_strings_only(rj.get(k), max_items=200))

    if not reqs:
        reqs = _extract_strings_only(rj, max_items=300)

    cleaned: List[str] = []
    for r in reqs:
        rr = _normalize_ws(r)
        if not rr:
            continue
        if rr.lower() in _CONTAINER_KEYS:
            continue
        if len(rr) < 2:
            continue
        if len(rr) > 180:
            rr = rr[:180].rstrip()
        cleaned.append(rr)

    seen = set()
    out: List[str] = []
    for r in cleaned:
        rl = r.lower()
        if rl in seen:
            continue
        seen.add(rl)
        out.append(r)

    return out[:60]


def _resume_years_max(text: str) -> Optional[int]:
    t = (text or "").lower()
    candidates: List[int] = []

    for m in re.finditer(r"\b(\d{1,2})\s*\+?\s*(?:years|yrs|year)\b", t):
        try:
            candidates.append(int(m.group(1)))
        except Exception:
            pass

    for m in re.finditer(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(?:years|yrs|year)\b", t):
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
    m = re.search(r"\b(\d{1,2})\s*\+\s*(?:years|yrs|year)\b", r)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\b(\d{1,2})\s*(?:years|yrs|year)\b", r)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _match_experience_requirement(req: str, resume_text: str, profile_strings: List[str]) -> Optional[bool]:
    rl = (req or "").lower()
    if "year" not in rl and "yrs" not in rl:
        return None
    if "experience" not in rl and "exp" not in rl and "backend" not in rl and "developer" not in rl and "development" not in rl:
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
    r = _normalize_ws(req)
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


def _job_as_text(job: Job) -> str:
    reqs = _extract_requirements(job.requirements_json or {})
    return "\n\n".join(
        [
            f"TITLE: {job.title}",
            job.description or "",
            "REQUIREMENTS:",
            "\n".join(reqs),
        ]
    ).strip()


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
    chunks = _split_into_chunks(_job_as_text(job))
    return len(chunks)


async def index_resume(db: AsyncSession, document_id: str) -> int:
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise RuntimeError(f"Document {document_id} not found")
    if not doc.raw_text:
        raise RuntimeError("document raw_text not ready")
    chunks = _split_into_chunks((doc.raw_text or "").strip())
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

    resume_text = (doc.raw_text or "").strip()
    resume_chunks = _split_into_chunks(resume_text)
    if not resume_chunks:
        return []

    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(resume_chunks)

    queries = _extract_requirements(job.requirements_json or {})
    if job.title:
        queries = [job.title] + queries
    if job.description:
        queries = queries + [job.description[:900]]

    seen = set()
    uniq_queries: List[str] = []
    for q in queries:
        qq = _normalize_ws(q)
        if not qq:
            continue
        ql = qq.lower()
        if ql in seen:
            continue
        seen.add(ql)
        uniq_queries.append(qq)

    if not uniq_queries:
        return []

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
) -> Tuple[int, str, List[str], List[Dict[str, Any]]]:
    reqs = _extract_requirements(requirements_json or {})
    skill_strings = _profile_skill_strings(profile_json or {})

    resume_text = " ".join([(e.get("text") or "") for e in (evidence_chunks or []) if isinstance(e.get("text"), str)])
    resume_text = _normalize_ws(resume_text)

    hits: List[str] = []
    missing: List[str] = []

    if reqs:
        for r in reqs:
            if _match_requirement(r, skill_strings, evidence_chunks, resume_text):
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
        prompt_version="v0",
        model_version="local_tfidf",
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

    evidence_chunks = await retrieve_evidence(db, job_id=app.job_id, candidate_id=app.candidate_id, k=5)

    score, rationale, missing, evidence = await calculate_score(
        job.requirements_json or {},
        profile.profile_json,
        evidence_chunks,
    )

    await save_scoring(db, application_id, score, rationale, missing, evidence)
    await db.commit()

    return {
        "application_id": application_id,
        "score": int(score),
        "rationale": rationale,
        "missing_requirements": missing,
        "evidence_snippets": evidence,
    }
