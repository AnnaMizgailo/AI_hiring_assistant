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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
from pydantic import EmailStr
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Конфигурация JWT
# Захардкоженный ключ для локального прототипа
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Настройка Google OAuth (для входа, а не для календаря)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_AUTH_REDIRECT_URI = os.getenv("GOOGLE_AUTH_REDIRECT_URI", "http://127.0.0.1:8000/api/auth/google/callback")

config_data = {
    "GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID,
    "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET
}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)



from database import (
    init_db, get_db, Job, Candidate, Application, Document,
    Profile, RecruiterGoogle, Slot, RagRun, Recruiter
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

class RecruiterRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class RecruiterLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    recruiter_id: str
    email: str
    full_name: Optional[str]


class ScreeningAnswersIn(BaseModel):
    salary_expectation: Optional[str] = None
    work_format: Optional[str] = None
    reason: Optional[str] = None
    english_level: Optional[str] = None


# === Приложение FastAPI ===
app = FastAPI(title="HR Assistant Prototype")

# Middleware для сессий (нужен для OAuth)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Подключаем статику и шаблоны
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    await init_db()


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_recruiter(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_recruiter_id: Optional[str] = Header(None, alias="X-Recruiter-ID"),
    db: AsyncSession = Depends(get_db)
) -> str:
    """
    Извлекает ID рекрутера из:
    - JWT токена (Bearer) в заголовке Authorization (для веб-интерфейса)
    - или из заголовка X-Recruiter-ID (для обратной совместимости с ботом)
    """
    # Если используется старый заголовок (для бота)
    if x_recruiter_id:
        return x_recruiter_id

    # Иначе проверяем JWT
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not authorization:
        raise credentials_exception
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise credentials_exception
    except ValueError:
        raise credentials_exception

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        recruiter_id: str = payload.get("sub")
        if recruiter_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Опционально: проверить, что рекрутер существует в БД
    result = await db.execute(select(Recruiter).where(Recruiter.id == recruiter_id))
    recruiter = result.scalar_one_or_none()
    if not recruiter:
        raise credentials_exception

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
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
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
    job.updated_at = datetime.utcnow()

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
            created_at=datetime.utcnow()
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
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
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
    app_obj.updated_at = datetime.utcnow()
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

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(recruiter: RecruiterRegister, db: AsyncSession = Depends(get_db)):
    # Проверяем, не занят ли email
    result = await db.execute(select(Recruiter).where(Recruiter.email == recruiter.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Создаём нового рекрутера
    recruiter_id = str(uuid.uuid4())
    hashed_password = get_password_hash(recruiter.password)
    db_recruiter = Recruiter(
        id=recruiter_id,
        email=recruiter.email,
        hashed_password=hashed_password,
        full_name=recruiter.full_name,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(db_recruiter)
    await db.commit()

    # Создаём токен
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": recruiter_id}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "recruiter_id": recruiter_id,
        "email": recruiter.email,
        "full_name": recruiter.full_name
    }

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(login_data: RecruiterLogin, db: AsyncSession = Depends(get_db)):
    # Ищем рекрутера по email
    result = await db.execute(select(Recruiter).where(Recruiter.email == login_data.email))
    recruiter = result.scalar_one_or_none()
    if not recruiter or not verify_password(login_data.password, recruiter.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    # Создаём токен
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": recruiter.id}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "recruiter_id": recruiter.id,
        "email": recruiter.email,
        "full_name": recruiter.full_name
    }

    
@app.get("/api/auth/google/login")
async def google_login(request: Request):
    # Теперь мы ссылаемся на УНИКАЛЬНОЕ имя функции
    redirect_uri = request.url_for('auth_google_callback')
    logger.info(f"Initiating Google Login. Redirect URI resolved to: {redirect_uri}")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/api/auth/google/callback")
async def auth_google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    logger.info("Entered auth_google_callback endpoint")
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            logger.error("Failed to get user_info from Google token")
            raise HTTPException(status_code=400, detail="Could not get user info from Google")

        email = user_info.get('email')
        google_id = user_info.get('sub')
        full_name = user_info.get('name')
        avatar_url = user_info.get('picture')

        logger.info(f"Successfully fetched Google user: {email} (ID: {google_id})")

        if not email or not google_id:
            logger.error(f"Missing essential fields: email={email}, sub={google_id}")
            raise HTTPException(status_code=400, detail="Missing required user info from Google")

        # Ищем существующего рекрутера по email или google_id
        result = await db.execute(
            select(Recruiter).where(
                (Recruiter.email == email) | (Recruiter.google_id == google_id)
            )
        )
        recruiter = result.scalar_one_or_none()

        if recruiter:
            logger.info(f"Found existing recruiter in DB: {recruiter.id}")
            if not recruiter.google_id:
                recruiter.google_id = google_id
            if not recruiter.avatar_url and avatar_url:
                recruiter.avatar_url = avatar_url
            recruiter.updated_at = datetime.utcnow()
        else:
            logger.info("Creating new recruiter in DB")
            recruiter_id = str(uuid.uuid4())
            recruiter = Recruiter(
                id=recruiter_id,
                email=email,
                google_id=google_id,
                full_name=full_name,
                avatar_url=avatar_url,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(recruiter)

        await db.commit()
        logger.info("DB commit successful")

        # Создаём JWT токен
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": recruiter.id}, expires_delta=access_token_expires
        )
        logger.info(f"Generated JWT token for recruiter: {recruiter.id}")

        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Вход выполнен</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .message {{ text-align: center; }}
            </style>
        </head>
        <body>
            <div class="message">
                <h1>✅ Вход выполнен успешно!</h1>
                <p>Вы можете закрыть это окно и вернуться в приложение.</p>
            </div>
            <script>
                if (window.opener) {{
                    console.log("Sending message to opener...");
                    window.opener.postMessage({{
                        type: 'google-auth-success',
                        token: '{access_token}',
                        recruiter_id: '{recruiter.id}',
                        email: '{email}',
                        full_name: '{full_name or ""}'
                    }}, '*');
                    setTimeout(() => window.close(), 2000);
                }} else {{
                    console.log("No window.opener found. Cannot pass token automatically.");
                }}
            </script>
        </body>
        </html>
        """)

    except Exception as e:
        logger.exception("Error occurred during Google Callback processing:")
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ошибка входа</title>
        </head>
        <body>
            <h1>Произошла ошибка при входе</h1>
            <p>{str(e)}</p>
        </body>
        </html>
        """)