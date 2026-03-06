from typing import Any, Dict, List, Optional, Tuple
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
from rag_text_utils import clean_text, keyword_coverage, normalize, skillish_keywords, split_into_chunks


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

_BOILERPLATE_PATTERNS = [
    r"\bemail\b",
    r"\bтелефон\b",
    r"\bгород\b",
    r"\bанглийский\b",
    r"\benglish\b",
    r"\bb1\b",
    r"\bb2\b",
    r"\bc1\b",
    r"\ba1\b",
    r"\ba2\b",
    r"\bожидания\b",
    r"\bзаметка\b",
    r"\bдополнительно\b",
    r"\bкратко о себе\b",
]

_CANON_MAP = {
    "react": ["react", "reactjs", "react.js"],
    "typescript": ["typescript", "ts", "type script"],
    "nextjs": ["nextjs", "next.js", "next js"],
    "git": ["git", "github", "gitlab"],
    "rest": ["rest", "rest api", "api"],
    "docker": ["docker"],
    "cicd": ["cicd", "ci/cd", "ci cd", "ci", "continuous integration", "code review", "код ревью", "ревью"],
    "python": ["python", "django", "flask", "fastapi", "celery"],
    "fastapi": ["fastapi"],
    "postgres": ["postgres", "postgresql", "postgre sql"],
    "redis": ["redis"],
    "django": ["django"],
    "sql": ["sql", "postgres", "postgresql", "mysql", "clickhouse", "bigquery"],
    "pandas": ["pandas"],
    "tableau": ["tableau", "power bi", "powerbi"],
    "abtesting": ["abtesting", "a/b testing", "ab testing", "experiments", "эксперименты"],
    "airflow": ["airflow"],
    "statistics": ["statistics", "статистика", "statistical"],
    "featureengineering": ["feature engineering", "featureengineering", "feature store"],
    "spark": ["spark", "pyspark"],
    "mlflow": ["mlflow"],
    "playwright": ["playwright"],
    "selenium": ["selenium"],
    "testdesign": ["test design", "test cases", "test scenarios", "negative cases", "контрактные проверки"],
    "productmanagement": ["product management", "product manager", "product owner", "pm"],
    "analytics": ["analytics", "analysis", "metrics", "dashboard", "dashboards", "insights", "sql", "tableau", "power bi"],
    "roadmap": ["roadmap", "backlog", "prioritization", "prioritisation", "priorities", "road map"],
    "jtbd": ["jtbd", "jobs to be done", "job to be done"],
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
        key = normalize(s)
        if key in seen:
            continue
        seen.add(key)
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

    must = _clean_requirement_list(
        _extract_strings_only(rj.get("must_have") or rj.get("must_haves") or [], max_items=200)
    )
    nice = _clean_requirement_list(
        _extract_strings_only(rj.get("nice_to_have") or rj.get("nice_to_haves") or [], max_items=200)
    )
    other = _clean_requirement_list(
        _extract_strings_only(rj.get("requirements") or [], max_items=200)
        + _extract_strings_only(rj.get("experience") or [], max_items=100)
        + _extract_strings_only(rj.get("languages") or [], max_items=100)
    )

    if not must and not nice and not other:
        must = _clean_requirement_list(_extract_strings_only(rj, max_items=300))

    return {
        "must_have": must,
        "nice_to_have": nice,
        "other": other,
        "all": _clean_requirement_list(must + nice + other, limit=120),
    }


def _profile_skill_strings(profile_json: Dict[str, Any]) -> List[str]:
    profile_json = profile_json or {}
    items: List[str] = []
    for key in ["skills", "technologies", "stack", "tools", "languages", "frameworks", "certifications", "summary", "projects", "experience", "education"]:
        if key in profile_json:
            items.extend(_extract_strings_only(profile_json.get(key), max_items=250))
    if not items:
        items = _extract_strings_only(profile_json, max_items=500)

    out: List[str] = []
    seen = set()
    for item in items:
        key = normalize(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _canonical_text(text: str) -> str:
    t = normalize(text)
    t = t.replace("next js", "nextjs").replace("next.js", "nextjs")
    t = t.replace("postgresql", "postgres").replace("postgre sql", "postgres")
    t = t.replace("ci/cd", "cicd").replace("ci cd", "cicd")
    t = t.replace("a/b testing", "abtesting").replace("ab testing", "abtesting")
    t = t.replace("job to be done", "jtbd").replace("jobs to be done", "jtbd")
    t = t.replace("type script", "typescript")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _canonical_requirement(req: str) -> str:
    c = _canonical_text(req)
    compact = re.sub(r"[^a-zа-я0-9]+", "", c)
    if compact in _CANON_MAP:
        return compact
    for key, aliases in _CANON_MAP.items():
        for alias in aliases:
            alias_compact = re.sub(r"[^a-zа-я0-9]+", "", _canonical_text(alias))
            if compact == alias_compact:
                return key
    return compact or c


def _aliases_for_requirement(req: str) -> List[str]:
    key = _canonical_requirement(req)
    aliases = _CANON_MAP.get(key)
    if aliases:
        return list({_canonical_text(x) for x in aliases + [req]})
    return [_canonical_text(req)]


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=1,
    )


def _best_matches(query: str, vectorizer: TfidfVectorizer, matrix, chunks: List[str], k: int) -> List[Tuple[float, str]]:
    if not query or not chunks:
        return []
    qv = vectorizer.transform([clean_text(query)])
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


def _sentence_split(text: str) -> List[str]:
    t = clean_text(text)
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", t)
    return [clean_text(x) for x in parts if clean_text(x)]


def _is_boilerplate(text: str) -> bool:
    tl = _canonical_text(text)
    if len(tl) < 20:
        return True
    for pat in _BOILERPLATE_PATTERNS:
        if re.search(pat, tl):
            return True
    return False


def _tech_density(text: str) -> float:
    tokens = re.findall(r"[a-zA-Zа-яА-Я0-9+#./-]{2,}", text or "")
    if not tokens:
        return 0.0
    tech_hits = 0
    canon_text = _canonical_text(text)
    for key, aliases in _CANON_MAP.items():
        if any(alias in canon_text for alias in aliases):
            tech_hits += 1
    return min(1.0, tech_hits / max(1, len(tokens) / 4.0))


def _best_display_snippet(chunk_text: str, query: str) -> str:
    query_aliases = _aliases_for_requirement(query)[:6]
    sentences = _sentence_split(chunk_text)
    if not sentences:
        return ""

    best = ""
    best_score = -999.0

    for sent in sentences:
        s = clean_text(sent)
        if not s:
            continue
        st = _canonical_text(s)
        alias_hits = sum(1 for a in query_aliases if a and a in st)
        score = alias_hits * 1.5 + _tech_density(s)
        if _is_boilerplate(s):
            score -= 1.0
        if len(s) < 20:
            score -= 1.0
        if score > best_score:
            best_score = score
            best = s

    if best_score < 0:
        return ""

    if len(best) > 260:
        best = best[:260].rstrip() + "..."
    return best


def _query_bonus(text: str, query: str) -> float:
    st = _canonical_text(text)
    aliases = _aliases_for_requirement(query)
    hits = sum(1 for a in aliases if a and a in st)
    return 0.04 * hits + 0.1 * _tech_density(text)


def _requirement_query_list(job: Job, req_groups: Dict[str, List[str]]) -> List[str]:
    title_queries = [clean_text(job.title or "")]
    must = req_groups.get("must_have", [])
    nice = req_groups.get("nice_to_have", [])
    title_tokens = skillish_keywords([job.title or ""], limit=12)
    out = title_queries + must + nice + title_tokens
    seen = set()
    final: List[str] = []
    for item in out:
        value = clean_text(item)
        if not value:
            continue
        key = _canonical_text(value)
        if key in seen:
            continue
        seen.add(key)
        final.append(value)
    return final[:32]


def _text_blob(profile_strings: List[str], evidence_chunks: List[Dict[str, Any]], resume_text: str) -> str:
    parts = list(profile_strings)
    parts.extend([clean_text(str(x.get("text") or "")) for x in evidence_chunks or []])
    parts.append(clean_text(resume_text or ""))
    return _canonical_text("\n".join([p for p in parts if p]))


def _match_requirement(req: str, profile_strings: List[str], evidence_chunks: List[Dict[str, Any]], resume_text: str) -> bool:
    blob = _text_blob(profile_strings, evidence_chunks, resume_text)
    aliases = _aliases_for_requirement(req)

    for alias in aliases:
        if alias and alias in blob:
            return True

    key = _canonical_requirement(req)

    if key == "productmanagement":
        return any(x in blob for x in ["product manager", "product owner", "pm ", "pm,", "pm.", "backlog", "roadmap"])
    if key == "analytics":
        return any(x in blob for x in ["analytics", "metrics", "dashboard", "dashboards", "insights", "sql", "tableau", "power bi"])
    if key == "roadmap":
        return any(x in blob for x in ["roadmap", "backlog", "prioritiz", "prioritis"])
    if key == "jtbd":
        return any(x in blob for x in ["jtbd", "jobs to be done", "job to be done", "user interview", "customer interview"])
    if key == "testdesign":
        return any(x in blob for x in ["test design", "test cases", "test scenarios", "negative cases", "contract"])
    return False


def _format_evidence(scored_chunks: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ev in enumerate(scored_chunks[:limit], start=1):
        out.append(
            {
                "text": clean_text(str(ev.get("text") or "")),
                "score": float(ev.get("score") or 0.0),
                "query": clean_text(str(ev.get("query") or "")),
                "why": clean_text(str(ev.get("why") or "")),
                "rank": i,
            }
        )
    return out


def _deterministic_rationale(
    must_total: int,
    must_hits: int,
    nice_total: int,
    nice_hits: int,
    other_total: int,
    other_hits: int,
    final_missing: List[str],
) -> str:
    parts = [
        f"Must-have coverage: {must_hits}/{must_total}" if must_total else "Must-have coverage: n/a",
        f"Nice-to-have coverage: {nice_hits}/{nice_total}" if nice_total else "Nice-to-have coverage: n/a",
        f"Other requirement coverage: {other_hits}/{other_total}" if other_total else "Other requirement coverage: n/a",
    ]
    if final_missing:
        parts.append("Missing: " + ", ".join(final_missing[:8]))
    else:
        parts.append("No validated missing requirements")
    return ". ".join(parts)


def _max_adjustment_for_score(score: int) -> int:
    if score < 30:
        return 3
    if score <= 60:
        return 5
    return 7


def _apply_score_guards(
    score: int,
    must_total: int,
    must_hits: int,
    nice_total: int,
    nice_hits: int,
    final_missing: List[str],
) -> int:
    out = int(score)
    must_cov = float(must_hits) / float(must_total) if must_total else 1.0
    nice_cov = float(nice_hits) / float(nice_total) if nice_total else 0.0

    if must_total > 0 and must_hits == 0:
        out = min(out, 20)
    if must_total > 0 and must_cov >= 0.8:
        out = max(out, 50)
    if must_total > 0 and must_cov == 1.0 and len(final_missing) == 0:
        out = max(out, 85)
    if must_total > 0 and must_cov >= 0.6 and len(final_missing) <= 2:
        out = max(out, 45)
    if must_total == 0 and len(final_missing) == 0:
        out = max(out, 70)
    if must_total > 0 and must_cov < 0.35:
        out = min(out, 45)
    if must_total > 0 and must_cov < 0.2 and nice_cov == 0:
        out = min(out, 25)

    return max(0, min(100, out))


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
    req_groups = _extract_requirement_groups(job.requirements_json or {})
    queries = _requirement_query_list(job, req_groups)

    scored: Dict[str, Dict[str, Any]] = {}

    for query in queries:
        matches = _best_matches(query, vectorizer, matrix, resume_chunks, k=max(3, min(8, k * 2)))
        for raw_score, chunk in matches:
            snippet = _best_display_snippet(chunk, query)
            if not snippet:
                continue
            if _is_boilerplate(snippet) and _tech_density(snippet) < 0.2:
                continue
            final_score = float(raw_score) + _query_bonus(snippet, query)
            key = _canonical_text(snippet)
            cur = scored.get(key)
            if cur is None or final_score > float(cur["score"]):
                scored[key] = {
                    "text": snippet,
                    "score": final_score,
                    "query": query,
                    "why": f"matched query: {query}",
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
) -> Tuple[int, str, List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    req_groups = _extract_requirement_groups(requirements_json or {})
    must_reqs = req_groups.get("must_have", [])
    nice_reqs = req_groups.get("nice_to_have", [])
    other_reqs = req_groups.get("other", [])
    all_reqs = req_groups.get("all", [])

    profile_strings = _profile_skill_strings(profile_json or {})
    resume_text_clean = clean_text(resume_text or "")

    must_hits: List[str] = []
    must_missing: List[str] = []
    nice_hits: List[str] = []
    nice_missing: List[str] = []
    other_hits: List[str] = []
    other_missing: List[str] = []

    for req in must_reqs:
        if _match_requirement(req, profile_strings, evidence_chunks, resume_text_clean):
            must_hits.append(req)
        else:
            must_missing.append(req)

    for req in nice_reqs:
        if _match_requirement(req, profile_strings, evidence_chunks, resume_text_clean):
            nice_hits.append(req)
        else:
            nice_missing.append(req)

    for req in other_reqs:
        if _match_requirement(req, profile_strings, evidence_chunks, resume_text_clean):
            other_hits.append(req)
        else:
            other_missing.append(req)

    must_cov = float(len(must_hits)) / float(len(must_reqs)) if must_reqs else 1.0
    nice_cov = float(len(nice_hits)) / float(len(nice_reqs)) if nice_reqs else 0.0
    other_cov = float(len(other_hits)) / float(len(other_reqs)) if other_reqs else 0.0
    evidence_strength = float(np.mean([float(x.get("score") or 0.0) for x in evidence_chunks[:5]])) if evidence_chunks else 0.0

    base_score = 100.0 * (0.75 * must_cov + 0.20 * nice_cov + 0.05 * other_cov)
    base_score += min(5.0, evidence_strength * 10.0)
    deterministic_score = int(round(max(0.0, min(100.0, base_score))))

    required_keywords = skillish_keywords(all_reqs, limit=48)
    keyword_coverage_percent = keyword_coverage(resume_text_clean, required_keywords)

    final_missing = must_missing + nice_missing + other_missing
    final_missing = [x for i, x in enumerate(final_missing) if normalize(x) not in {normalize(y) for y in final_missing[:i]}]

    evidence_out = _format_evidence(evidence_chunks, limit=12)
    rationale_body = _deterministic_rationale(
        must_total=len(must_reqs),
        must_hits=len(must_hits),
        nice_total=len(nice_reqs),
        nice_hits=len(nice_hits),
        other_total=len(other_reqs),
        other_hits=len(other_hits),
        final_missing=final_missing,
    )

    llm_adjustment = 0
    llm_missing_raw: List[str] = []
    scoring_mode = "FALLBACK"

    deterministic_facts = {
        "must_have": must_reqs,
        "nice_to_have": nice_reqs,
        "must_hits": must_hits,
        "must_missing": must_missing,
        "nice_hits": nice_hits,
        "nice_missing": nice_missing,
        "other_hits": other_hits,
        "other_missing": other_missing,
        "deterministic_score": deterministic_score,
        "keyword_coverage_percent": keyword_coverage_percent,
    }

    if evidence_chunks:
        try:
            llm = await asyncio.to_thread(
                llm_score_application,
                job_title,
                deterministic_facts,
                evidence_out,
                None,
                _max_adjustment_for_score(deterministic_score),
            )
        except Exception:
            llm = None

        if llm is not None:
            scoring_mode = "LLM"
            llm_adjustment = int(llm.get("score_adjustment") or 0)
            llm_missing_raw = [x for x in llm.get("focus_missing") or [] if normalize(x) in {normalize(y) for y in final_missing}]
            llm_rationale = clean_text(str(llm.get("rationale") or ""))
            if llm_rationale:
                rationale_body = llm_rationale

    final_score = _apply_score_guards(
        deterministic_score + llm_adjustment,
        must_total=len(must_reqs),
        must_hits=len(must_hits),
        nice_total=len(nice_reqs),
        nice_hits=len(nice_hits),
        final_missing=final_missing,
    )

    rationale = f"[{scoring_mode}] {rationale_body}"

    debug = {
        "deterministic_score": deterministic_score,
        "llm_adjustment": llm_adjustment,
        "must_have_total": len(must_reqs),
        "must_have_hits": len(must_hits),
        "nice_to_have_total": len(nice_reqs),
        "nice_to_have_hits": len(nice_hits),
        "other_total": len(other_reqs),
        "other_hits": len(other_hits),
        "keyword_coverage_percent": keyword_coverage_percent,
        "deterministic_missing": must_missing + nice_missing + other_missing,
        "llm_missing_raw": llm_missing_raw,
        "final_missing": final_missing,
    }

    return final_score, rationale, final_missing, evidence_out, scoring_mode, debug


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
    evidence_chunks = await retrieve_evidence(db, app.job_id, app.candidate_id, k=5)

    score, rationale, missing, evidence, scoring_mode, debug = await calculate_score(
        requirements_json=job.requirements_json or {},
        profile_json=profile.profile_json or {},
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
    )

    await db.commit()

    result = {
        "application_id": application_id,
        "score": int(score),
        "rationale": rationale,
        "missing_requirements": missing,
        "evidence_snippets": evidence,
        "scoring_mode": scoring_mode,
    }
    result.update(debug)
    return result