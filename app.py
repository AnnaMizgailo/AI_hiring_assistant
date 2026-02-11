from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import uuid

from parsing import save_resume_file, parse_document
from scoring_rag import score_application
from calendar_llm import get_google_oauth_url, handle_google_callback, get_available_slots, book_slot

class JobCreate(BaseModel):
    title: str
    description: str
    threshold_score: int = Field(ge=0, le=100)
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

class MemoryStore:
    def __init__(self) -> None:
        self.users: Dict[str, Dict[str, Any]] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.telegram_to_candidate: Dict[int, str] = {}
        self.applications: Dict[str, Dict[str, Any]] = {}
        self.recruiter_google: Dict[str, Dict[str, Any]] = {}
        self.slots: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}

STORE = MemoryStore()

def now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def get_store() -> MemoryStore:
    return STORE

def ensure_candidate(store: MemoryStore, telegram_user_id: int, telegram_username: str | None) -> str:
    if telegram_user_id in store.telegram_to_candidate:
        return store.telegram_to_candidate[telegram_user_id]
    cid = str(uuid.uuid4())
    store.candidates[cid] = {
        "id": cid,
        "telegram_user_id": telegram_user_id,
        "telegram_username": telegram_username,
        "full_name": None,
        "contacts_json": None,
        "created_at": now_iso(),
    }
    store.telegram_to_candidate[telegram_user_id] = cid
    return cid

app = FastAPI(title="HR Assistant Prototype")

@app.post("/api/jobs", response_model=Dict[str, str])
def create_job(payload: JobCreate, store: MemoryStore = Depends(get_store)):
    jid = str(uuid.uuid4())
    store.jobs[jid] = {
        "id": jid,
        "title": payload.title,
        "description": payload.description,
        "threshold_score": payload.threshold_score,
        "requirements_json": payload.requirements_json or {},
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "recruiter_id": "demo-recruiter",
    }
    return {"job_id": jid}

@app.get("/api/jobs", response_model=List[JobOut])
def list_jobs(store: MemoryStore = Depends(get_store)):
    return [
        {"id": j["id"], "title": j["title"], "threshold_score": j["threshold_score"], "created_at": j["created_at"]}
        for j in store.jobs.values()
    ]

@app.get("/api/jobs/{job_id}", response_model=Dict[str, Any])
def get_job(job_id: str, store: MemoryStore = Depends(get_store)):
    j = store.jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return j

@app.post("/api/applications", response_model=CreateApplicationOut)
def create_application(payload: CreateApplicationIn, store: MemoryStore = Depends(get_store)):
    if payload.job_id not in store.jobs:
        raise HTTPException(status_code=404, detail="job not found")
    candidate_id = ensure_candidate(store, payload.telegram_user_id, payload.telegram_username)
    application_id = str(uuid.uuid4())
    store.applications[application_id] = {
        "id": application_id,
        "job_id": payload.job_id,
        "candidate_id": candidate_id,
        "status": "NEW",
        "score": None,
        "score_rationale": None,
        "missing_requirements_json": None,
        "evidence_snippets_json": None,
        "screening_answers_json": {},
        "screening_summary": None,
        "calendar_event_id": None,
        "feedback_text": None,
        "last_error": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    return {"application_id": application_id, "candidate_id": candidate_id, "status": "NEW"}

@app.get("/api/jobs/{job_id}/applications", response_model=List[ApplicationOut])
def list_applications(job_id: str, store: MemoryStore = Depends(get_store)):
    if job_id not in store.jobs:
        raise HTTPException(status_code=404, detail="job not found")
    out = []
    for a in store.applications.values():
        if a["job_id"] != job_id:
            continue
        c = store.candidates.get(a["candidate_id"], {})
        out.append(
            {
                "application_id": a["id"],
                "candidate_id": a["candidate_id"],
                "candidate_name": c.get("full_name"),
                "score": a.get("score"),
                "status": a.get("status"),
                "screening_summary": a.get("screening_summary"),
                "created_at": a.get("created_at"),
            }
        )
    return sorted(out, key=lambda x: (x["score"] is None, -(x["score"] or 0)))

@app.post("/api/candidates/{candidate_id}/documents", response_model=Dict[str, str])
async def upload_document(candidate_id: str, file: UploadFile = File(...), store: MemoryStore = Depends(get_store)):
    if candidate_id not in store.candidates:
        raise HTTPException(status_code=404, detail="candidate not found")
    file_bytes = await file.read()
    doc = save_resume_file(store=store, candidate_id=candidate_id, file_bytes=file_bytes, filename=file.filename, mime_type=file.content_type or "")
    return {"document_id": doc["document_id"]}

@app.post("/api/parsing/run", response_model=Dict[str, Any])
def run_parsing(document_id: str, store: MemoryStore = Depends(get_store)):
    if document_id not in store.documents:
        raise HTTPException(status_code=404, detail="document not found")
    res = parse_document(store=store, document_id=document_id)
    return res

@app.post("/api/applications/{application_id}/score", response_model=ScoreOut)
def run_scoring(application_id: str, store: MemoryStore = Depends(get_store)):
    if application_id not in store.applications:
        raise HTTPException(status_code=404, detail="application not found")
    res = score_application(store=store, application_id=application_id)
    return res

@app.get("/api/google/oauth/url", response_model=GoogleOauthOut)
def google_oauth_url(store: MemoryStore = Depends(get_store)):
    auth_url = get_google_oauth_url(store=store, recruiter_id="demo-recruiter")
    return {"auth_url": auth_url}

@app.get("/api/google/oauth/callback", response_model=GoogleCallbackOut)
def google_oauth_callback(code: str, store: MemoryStore = Depends(get_store)):
    handle_google_callback(store=store, recruiter_id="demo-recruiter", code=code)
    return {"status": "OK"}

@app.get("/api/jobs/{job_id}/slots", response_model=SlotsOut)
def list_slots(job_id: str, store: MemoryStore = Depends(get_store)):
    if job_id not in store.jobs:
        raise HTTPException(status_code=404, detail="job not found")
    slots = get_available_slots(store=store, job_id=job_id, recruiter_id="demo-recruiter")
    return {"slots": slots}

@app.post("/api/applications/{application_id}/book-slot", response_model=BookSlotOut)
def api_book_slot(application_id: str, payload: BookSlotIn, store: MemoryStore = Depends(get_store)):
    if application_id not in store.applications:
        raise HTTPException(status_code=404, detail="application not found")
    res = book_slot(store=store, application_id=application_id, slot_id=payload.slot_id)
    return res
