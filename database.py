import datetime
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, JSON, DateTime, Text, BigInteger, ForeignKey

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./hr_assistant.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    threshold_score = Column(Integer, nullable=False)
    requirements_json = Column(JSON, default={})
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    recruiter_id = Column(String, nullable=False)


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(String, primary_key=True)
    telegram_user_id = Column(BigInteger, unique=True, nullable=False)
    telegram_username = Column(String, nullable=True)
    full_name = Column(String, nullable=True)
    contacts_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class Application(Base):
    __tablename__ = "applications"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    candidate_id = Column(String, ForeignKey("candidates.id"), nullable=False)
    status = Column(String, default="NEW")
    score = Column(Integer, nullable=True)
    score_rationale = Column(Text, nullable=True)
    missing_requirements_json = Column(JSON, nullable=True)
    evidence_snippets_json = Column(JSON, nullable=True)
    screening_answers_json = Column(JSON, default={})
    screening_summary = Column(Text, nullable=True)
    calendar_event_id = Column(String, nullable=True)
    scheduled_start_dt = Column(DateTime, nullable=True)
    scheduled_end_dt = Column(DateTime, nullable=True)
    feedback_text = Column(Text, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    candidate_id = Column(String, ForeignKey("candidates.id"), nullable=False)
    file_name = Column(String)
    mime_type = Column(String)
    file_path = Column(String)
    file_hash = Column(String, nullable=True)
    raw_text = Column(Text, nullable=True)
    parse_status = Column(String, default="PENDING")
    last_error = Column(Text, nullable=True)
    parsed_at = Column(DateTime, nullable=True)


class Profile(Base):
    __tablename__ = "profiles"

    candidate_id = Column(String, ForeignKey("candidates.id"), primary_key=True)
    profile_json = Column(JSON)
    confidence_json = Column(JSON)
    missing_fields_json = Column(JSON)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class RecruiterGoogle(Base):
    __tablename__ = "recruiter_google"

    recruiter_id = Column(String, primary_key=True)
    access_token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    token_expiry = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Slot(Base):
    __tablename__ = "slots"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    recruiter_id = Column(String, nullable=True)
    application_id = Column(String, ForeignKey("applications.id"), nullable=True)
    start_dt = Column(DateTime, nullable=True)
    end_dt = Column(DateTime, nullable=True)
    label = Column(String, nullable=True)
    status = Column(String, default="FREE")
    google_event_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class RagRun(Base):
    __tablename__ = "rag_runs"

    id = Column(String, primary_key=True)
    application_id = Column(String, ForeignKey("applications.id"))
    top_k_chunks_json = Column(JSON)
    prompt_version = Column(String)
    model_version = Column(String)
    created_at = Column(DateTime, default=utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
