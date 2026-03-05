from typing import Any, Dict, List, Tuple
import uuid
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import Application, Job, Profile, RagRun

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def _utcnow() -> datetime.datetime:
    """Возвращает текущее UTC время как naive datetime (без часового пояса)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

async def index_job(db: AsyncSession, job_id: str) -> int:
    raise NotImplementedError()

async def index_resume(db: AsyncSession, document_id: str) -> int:
    raise NotImplementedError()

async def retrieve_evidence(db: AsyncSession, job_id: str, candidate_id: str, k: int = 5) -> List[Dict[str, Any]]:
    raise NotImplementedError()

async def calculate_score(requirements_json: Dict[str, Any], profile_json: Dict[str, Any], evidence_chunks: List[Dict[str, Any]]) -> Tuple[int, str, List[str], List[Dict[str, Any]]]:
    raise NotImplementedError()

async def save_scoring(db: AsyncSession, application_id: str, score: int, rationale: str, missing: List[str], evidence: List[Dict[str, Any]]) -> None:
    app_result = await db.execute(
        select(Application).where(Application.id == application_id)
    )
    app = app_result.scalar_one()
    app.score = score
    app.score_rationale = rationale
    app.missing_requirements_json = missing
    app.evidence_snippets_json = evidence
    app.status = "SCORING_DONE"
    app.updated_at = _utcnow()

    # Создаём запись о запуске RAG
    rag_run = RagRun(
        id=str(uuid.uuid4()),
        application_id=application_id,
        top_k_chunks_json=evidence,  # предположительно evidence_chunks
        prompt_version="v0",
        model_version="unset",
        created_at=_utcnow()
    )
    db.add(rag_run)

async def score_application(db: AsyncSession, application_id: str) -> Dict[str, Any]:
    app_result = await db.execute(
        select(Application).where(Application.id == application_id)
    )
    app = app_result.scalar_one_or_none()
    if not app:
        raise RuntimeError(f"Application {application_id} not found")

    # Получаем вакансию
    job_result = await db.execute(
        select(Job).where(Job.id == app.job_id)
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise RuntimeError(f"Job {app.job_id} not found")

    # Получаем профиль кандидата
    profile_result = await db.execute(
        select(Profile).where(Profile.candidate_id == app.candidate_id)
    )
    profile = profile_result.scalar_one_or_none()
    if not profile:
        raise RuntimeError("candidate profile not ready")

    requirements_json = job.requirements_json or {}
    evidence_chunks = await retrieve_evidence(
        db,
        job_id=app.job_id,
        candidate_id=app.candidate_id,
        k=5
    )

    score, rationale, missing, evidence = await calculate_score(
        requirements_json,
        profile.profile_json,
        evidence_chunks
    )

    await save_scoring(
        db,
        application_id,
        score,
        rationale,
        missing,
        evidence
    )

    # Сохраняем все изменения
    await db.commit()

    return {
        "application_id": application_id,
        "score": int(score),
        "rationale": rationale,
        "missing_requirements": missing,
        "evidence_snippets": evidence,
    }