from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import uuid
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from database import (
    init_db, get_db, Job, Candidate, Application, Document,
    Profile, RecruiterGoogle, Slot, RagRun
)
from parsing import save_resume_file, parse_document
from scoring_rag import score_application
from calendar_llm import get_google_oauth_url, handle_google_callback, get_available_slots, book_slot


# === Pydantic модели ===
class JobCreate(BaseModel):
    title: str
    description: str
    threshold_score: int = Field(ge=0, le=100)
    requirements_json: Optional[Dict[str, Any]] = None


class JobUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    threshold_score: Optional[int] = Field(None, ge=0, le=100)
    requirements_json: Optional[Dict[str, Any]] = None


class JobOut(BaseModel):
    id: str
    title: str
    threshold_score: int
    created_at: str


class ApplicationOut(BaseModel):
    application_id: str
    candidate_id: str
    candidate_name: Optional[str] = None
    score: Optional[int] = None
    status: str
    screening_summary: Optional[str] = None
    created_at: str


class CreateApplicationIn(BaseModel):
    job_id: str
    telegram_user_id: int
    telegram_username: Optional[str] = None


class CreateApplicationOut(BaseModel):
    application_id: str
    candidate_id: str
    status: str


class ScoreOut(BaseModel):
    application_id: str
    score: int
    rationale: str
    missing_requirements: List[str]
    evidence_snippets: List[Dict[str, Any]]


class SlotsOut(BaseModel):
    slots: List[Dict[str, Any]]


class BookSlotIn(BaseModel):
    slot_id: str


class BookSlotOut(BaseModel):
    status: str
    event_id: str
    start_dt: str
    end_dt: str


class GoogleOauthOut(BaseModel):
    auth_url: str


class GoogleCallbackOut(BaseModel):
    status: str


# === Приложение FastAPI ===
app = FastAPI(title="HR Assistant Prototype")

# Подключаем статику и шаблоны
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# === Инициализация БД при старте ===
@app.on_event("startup")
async def startup_event():
    await init_db()


# === Вспомогательные функции ===
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


async def get_current_recruiter(recruiter_id: Optional[str] = Header(None, alias="X-Recruiter-ID")):
    if not recruiter_id:
        raise HTTPException(status_code=401, detail="Missing recruiter ID")
    return recruiter_id


# === Эндпоинты ===

@app.post("/api/jobs", response_model=Dict[str, str])
async def create_job(
        payload: JobCreate,
        recruiter: str = Depends(get_current_recruiter),
        db: AsyncSession = Depends(get_db)
):
    jid = str(uuid.uuid4())
    job = Job(
        id=jid,
        title=payload.title,
        description=payload.description,
        threshold_score=payload.threshold_score,
        requirements_json=payload.requirements_json or {},
        recruiter_id=recruiter,
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow()
    )
    db.add(job)
    await db.commit()
    return {"job_id": jid}


@app.get("/api/jobs", response_model=List[JobOut])
async def list_jobs(
        recruiter: str = Depends(get_current_recruiter),
        db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Job).where(Job.recruiter_id == recruiter)
    )
    jobs = result.scalars().all()
    return [
        {
            "id": j.id,
            "title": j.title,
            "threshold_score": j.threshold_score,
            "created_at": j.created_at.isoformat() + "Z"
        }
        for j in jobs
    ]


@app.get("/api/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job(
        job_id: str,
        recruiter: str = Depends(get_current_recruiter),
        db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "threshold_score": job.threshold_score,
        "requirements_json": job.requirements_json,
        "created_at": job.created_at.isoformat() + "Z",
        "updated_at": job.updated_at.isoformat() + "Z",
        "recruiter_id": job.recruiter_id
    }


@app.put("/api/jobs/{job_id}", response_model=Dict[str, str])
async def update_job(
        job_id: str,
        payload: JobUpdate,
        recruiter: str = Depends(get_current_recruiter),
        db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    update_data = payload.dict(exclude_unset=True)
    for key, value in update_data.items():
        if hasattr(job, key):
            setattr(job, key, value)
    job.updated_at = datetime.datetime.utcnow()

    await db.commit()
    return {"job_id": job_id, "status": "updated"}


@app.delete("/api/jobs/{job_id}", response_model=Dict[str, str])
async def delete_job(
        job_id: str,
        recruiter: str = Depends(get_current_recruiter),
        db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    await db.delete(job)
    await db.commit()
    return {"job_id": job_id, "status": "deleted"}


@app.post("/api/applications", response_model=CreateApplicationOut)
async def create_application(
        payload: CreateApplicationIn,
        db: AsyncSession = Depends(get_db)
):
    job_result = await db.execute(select(Job).where(Job.id == payload.job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    cand_result = await db.execute(
        select(Candidate).where(Candidate.telegram_user_id == payload.telegram_user_id)
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        candidate_id = str(uuid.uuid4())
        candidate = Candidate(
            id=candidate_id,
            telegram_user_id=payload.telegram_user_id,
            telegram_username=payload.telegram_username,
            created_at=datetime.datetime.utcnow()
        )
        db.add(candidate)
        await db.flush()
    else:
        candidate_id = candidate.id

    application_id = str(uuid.uuid4())
    app = Application(
        id=application_id,
        job_id=payload.job_id,
        candidate_id=candidate_id,
        status="NEW",
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow()
    )
    db.add(app)
    await db.commit()

    return {
        "application_id": application_id,
        "candidate_id": candidate_id,
        "status": "NEW"
    }


@app.get("/api/jobs/{job_id}/applications", response_model=List[ApplicationOut])
async def list_applications(
        job_id: str,
        status: Optional[str] = None,
        min_score: Optional[int] = Query(None, ge=0, le=100),
        max_score: Optional[int] = Query(None, ge=0, le=100),
        recruiter: str = Depends(get_current_recruiter),
        db: AsyncSession = Depends(get_db)
):
    job_result = await db.execute(
        select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter)
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    query = select(Application).where(Application.job_id == job_id)
    if status:
        query = query.where(Application.status == status)
    if min_score is not None:
        query = query.where(Application.score >= min_score)
    if max_score is not None:
        query = query.where(Application.score <= max_score)

    result = await db.execute(query)
    apps = result.scalars().all()

    out = []
    for app in apps:
        cand_result = await db.execute(select(Candidate).where(Candidate.id == app.candidate_id))
        candidate = cand_result.scalar_one_or_none()
        out.append({
            "application_id": app.id,
            "candidate_id": app.candidate_id,
            "candidate_name": candidate.full_name if candidate else None,
            "score": app.score,
            "status": app.status,
            "screening_summary": app.screening_summary,
            "created_at": app.created_at.isoformat() + "Z",
        })

    out.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))
    return out


@app.post("/api/candidates/{candidate_id}/documents", response_model=Dict[str, str])
async def upload_document(
        candidate_id: str,
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db)
):
    cand_result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="candidate not found")

    file_bytes = await file.read()

    # Теперь используем save_resume_file из parsing.py для чистоты кода
    result = await save_resume_file(
        db=db,
        candidate_id=candidate_id,
        file_bytes=file_bytes,
        filename=file.filename or "",
        mime_type=file.content_type or "",
    )
    await db.commit()

    return {"document_id": result["document_id"]}


@app.post("/api/parsing/run", response_model=Dict[str, Any])
async def run_parsing(
        document_id: str,
        db: AsyncSession = Depends(get_db)
):
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")

    res = await parse_document(db, document_id)
    return res


@app.post("/api/applications/{application_id}/score", response_model=ScoreOut)
async def run_scoring(
        application_id: str,
        db: AsyncSession = Depends(get_db)
):
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app = app_result.scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=404, detail="application not found")

    res = await score_application(db, application_id)
    return res


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse("jobs.html", {"request": request})


@app.get("/jobs/new", response_class=HTMLResponse)
async def new_job_page(request: Request):
    return templates.TemplateResponse("job_form.html", {"request": request, "job": None})


@app.get("/jobs/{job_id}/edit", response_class=HTMLResponse)
async def edit_job_page(request: Request, job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "threshold_score": job.threshold_score,
        "requirements_json": job.requirements_json,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "recruiter_id": job.recruiter_id
    }
    return templates.TemplateResponse("job_form.html", {"request": request, "job": job_dict})


@app.get("/jobs/{job_id}/applications", response_class=HTMLResponse)
async def applications_page(request: Request, job_id: str):
    return templates.TemplateResponse("applications.html", {"request": request, "job_id": job_id})