from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Query, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import uuid
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import (
    init_db, get_db, Job, Candidate, Application, Document,
    Profile, RecruiterGoogle, Slot, RagRun
)
from parsing import parse_document
from scoring_rag import score_application
from calendar_llm import (
    get_google_oauth_url,
    handle_google_callback,
    get_available_slots,
    book_slot,
    generate_screening_summary,
    generate_feedback,
)


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


class ParsingRunIn(BaseModel):
    document_id: str


class ScreeningAnswersIn(BaseModel):
    salary_expectation: Optional[str] = None
    work_format: Optional[str] = None
    reason: Optional[str] = None
    english_level: Optional[str] = None


# === Приложение FastAPI ===
app = FastAPI(title="HR Assistant Prototype")

# Подключаем статику и шаблоны
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    await init_db()


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


async def get_current_recruiter(recruiter_id: Optional[str] = Header(None, alias="X-Recruiter-ID")):
    if not recruiter_id:
        raise HTTPException(status_code=401, detail="Missing recruiter ID")
    return recruiter_id


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
    result = await db.execute(select(Job).where(Job.recruiter_id == recruiter))
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


@app.get("/api/public/jobs", response_model=List[JobOut])
async def list_public_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job))
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
    result = await db.execute(select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter))
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
    result = await db.execute(select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter))
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
    result = await db.execute(select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter))
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

    cand_result = await db.execute(select(Candidate).where(Candidate.telegram_user_id == payload.telegram_user_id))
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
    app_obj = Application(
        id=application_id,
        job_id=payload.job_id,
        candidate_id=candidate_id,
        status="NEW",
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow()
    )
    db.add(app_obj)
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
    job_result = await db.execute(select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter))
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
    for app_obj in apps:
        cand_result = await db.execute(select(Candidate).where(Candidate.id == app_obj.candidate_id))
        candidate = cand_result.scalar_one_or_none()
        out.append({
            "application_id": app_obj.id,
            "candidate_id": app_obj.candidate_id,
            "candidate_name": candidate.full_name if candidate else None,
            "score": app_obj.score,
            "status": app_obj.status,
            "screening_summary": app_obj.screening_summary,
            "created_at": app_obj.created_at.isoformat() + "Z",
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
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    storage_dir = os.path.join(os.getcwd(), "storage_resumes")
    os.makedirs(storage_dir, exist_ok=True)

    document_id = str(uuid.uuid4())
    safe_name = file.filename or f"{document_id}.bin"
    file_path = os.path.join(storage_dir, f"{document_id}__{safe_name}")

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    doc = Document(
        id=document_id,
        candidate_id=candidate_id,
        file_name=safe_name,
        mime_type=file.content_type or "",
        file_path=file_path,
        parse_status="PENDING"
    )
    db.add(doc)
    await db.commit()

    return {"document_id": document_id}


async def _parse_document_background(document_id: str):
    from database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        await parse_document(session, document_id)


@app.post("/api/parsing/run", response_model=Dict[str, Any])
async def run_parsing(
    payload: ParsingRunIn,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    document_id = payload.document_id
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")

    doc.parse_status = "PROCESSING"
    doc.last_error = None
    await db.commit()

    background_tasks.add_task(_parse_document_background, document_id)
    return {"status": "STARTED", "document_id": document_id, "parse_status": doc.parse_status}


@app.get("/api/documents/{document_id}", response_model=Dict[str, Any])
async def get_document_status(document_id: str, db: AsyncSession = Depends(get_db)):
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    return {
        "document_id": doc.id,
        "parse_status": doc.parse_status,
        "last_error": doc.last_error,
        "parsed_at": doc.parsed_at.isoformat() if doc.parsed_at else None,
    }


@app.patch("/api/applications/{application_id}/screening-answers", response_model=Dict[str, Any])
async def update_screening_answers(
    application_id: str,
    payload: ScreeningAnswersIn,
    db: AsyncSession = Depends(get_db),
):
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app_obj = app_result.scalar_one_or_none()
    if not app_obj:
        raise HTTPException(status_code=404, detail="application not found")

    app_obj.screening_answers_json = {
        "salary_expectation": payload.salary_expectation,
        "work_format": payload.work_format,
        "reason": payload.reason,
        "english_level": payload.english_level,
    }
    app_obj.updated_at = datetime.datetime.utcnow()
    await db.commit()
    return {"status": "ok", "application_id": application_id}


@app.post("/api/applications/{application_id}/score", response_model=ScoreOut)
async def run_scoring(
    application_id: str,
    db: AsyncSession = Depends(get_db)
):
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app_obj = app_result.scalar_one_or_none()
    if not app_obj:
        raise HTTPException(status_code=404, detail="application not found")

    res = await score_application(db, application_id)
    return res


@app.post("/api/applications/{application_id}/decision", response_model=Dict[str, Any])
async def make_decision(
    application_id: str,
    recruiter: Optional[str] = Header(None, alias="X-Recruiter-ID"),
    db: AsyncSession = Depends(get_db),
):
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app_obj = app_result.scalar_one_or_none()
    if not app_obj:
        raise HTTPException(status_code=404, detail="application not found")

    job_result = await db.execute(select(Job).where(Job.id == app_obj.job_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    effective_recruiter = recruiter or job.recruiter_id
    if recruiter and job.recruiter_id != recruiter:
        raise HTTPException(status_code=403, detail="forbidden")

    if app_obj.score is None:
        try:
            await score_application(db, application_id)
            await db.refresh(app_obj)
        except RuntimeError as exc:
            detail = str(exc)
            if "candidate profile not ready" in detail.lower():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Candidate profile is not ready yet. "
                        "Resume parsing has not completed successfully. "
                        "Check Ollama settings and document parse_status/last_error."
                    ),
                )
            raise HTTPException(status_code=500, detail=detail)

    summary = await generate_screening_summary(db, application_id)
    threshold = job.threshold_score or 0
    score = app_obj.score or 0

    if score >= threshold:
        slots = await get_available_slots(db, str(job.id), effective_recruiter)
        return {
            "decision": "INTERVIEW",
            "score": score,
            "threshold_score": threshold,
            "screening_summary": summary,
            "slots": slots,
        }
    else:
        feedback = await generate_feedback(db, application_id)
        return {
            "decision": "REJECT",
            "score": score,
            "threshold_score": threshold,
            "screening_summary": summary,
            "feedback": feedback,
        }


@app.get("/api/google/oauth-url", response_model=GoogleOauthOut)
async def google_oauth_url(
    recruiter: str = Depends(get_current_recruiter),
    db: AsyncSession = Depends(get_db),
):
    auth_url = await get_google_oauth_url(db, recruiter)
    return {"auth_url": auth_url}


@app.get("/api/google/callback", response_model=GoogleCallbackOut)
async def google_callback(
    code: str,
    state: Optional[str] = None,
    recruiter: Optional[str] = Header(None, alias="X-Recruiter-ID"),
    db: AsyncSession = Depends(get_db),
):
    recruiter_id = recruiter or state
    if not recruiter_id:
        raise HTTPException(status_code=400, detail="Missing recruiter ID")
    result = await handle_google_callback(db, recruiter_id, code)
    return {"status": result["status"]}


@app.get("/integrations", response_class=HTMLResponse)
async def integrations_page(request: Request):
    return templates.TemplateResponse(
        "integrations.html",
        {
            "request": request,
            "recruiter_id": os.getenv("RECRUITER_ID", "demo-recruiter"),
        },
    )


@app.get("/google/connect")
async def google_connect_page(
    db: AsyncSession = Depends(get_db),
):
    recruiter_id = os.getenv("RECRUITER_ID", "demo-recruiter")
    auth_url = await get_google_oauth_url(db, recruiter_id)
    return RedirectResponse(url=auth_url)


@app.post("/api/jobs/{job_id}/slots", response_model=SlotsOut)
async def create_slots(
    job_id: str,
    recruiter: str = Depends(get_current_recruiter),
    db: AsyncSession = Depends(get_db),
):
    slots = await get_available_slots(db, job_id, recruiter)
    return {"slots": slots}


@app.post("/api/applications/{application_id}/book-slot", response_model=BookSlotOut)
async def api_book_slot(
    application_id: str,
    payload: BookSlotIn,
    db: AsyncSession = Depends(get_db),
):
    return await book_slot(db, application_id, payload.slot_id)


@app.post("/api/applications/{application_id}/summary", response_model=Dict[str, str])
async def api_generate_summary(
    application_id: str,
    db: AsyncSession = Depends(get_db),
):
    summary = await generate_screening_summary(db, application_id)
    return {"screening_summary": summary}


@app.post("/api/applications/{application_id}/feedback", response_model=Dict[str, str])
async def api_generate_feedback(
    application_id: str,
    db: AsyncSession = Depends(get_db),
):
    feedback = await generate_feedback(db, application_id)
    return {"feedback": feedback}


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
        "recruiter_id": job.recruiter_id,
    }
    return templates.TemplateResponse("job_form.html", {"request": request, "job": job_dict})


@app.get("/jobs/{job_id}/applications", response_class=HTMLResponse)
async def applications_page(request: Request, job_id: str):
    return templates.TemplateResponse("applications.html", {"request": request, "job_id": job_id})