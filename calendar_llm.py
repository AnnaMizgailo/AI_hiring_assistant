from typing import Any, Dict, List
import uuid
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import RecruiterGoogle, Slot, Application, Job, Candidate

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

async def get_google_oauth_url(db: AsyncSession, recruiter_id: str) -> str:
    raise NotImplementedError()

async def handle_google_callback(db: AsyncSession, recruiter_id: str, code: str) -> Dict[str, Any]:
    raise NotImplementedError()

async def get_freebusy(db: AsyncSession, recruiter_id: str, time_min: str, time_max: str) -> List[Dict[str, Any]]:
    raise NotImplementedError()

async def build_candidate_slots(free_windows: List[Dict[str, Any]], slot_minutes: int = 30, max_slots: int = 8) -> List[Dict[str, Any]]:
    raise NotImplementedError()

async def create_calendar_event(db: AsyncSession, recruiter_id: str, start_dt: str, end_dt: str, candidate_name: str, description: str) -> Dict[str, Any]:
    raise NotImplementedError()

async def get_available_slots(db: AsyncSession, job_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError()

async def book_slot(db: AsyncSession, application_id: str, slot_id: str) -> Dict[str, Any]:
    raise NotImplementedError()

async def generate_screening_summary(db: AsyncSession, application_id: str) -> str:
    raise NotImplementedError()

async def generate_feedback(db: AsyncSession, application_id: str) -> str:
    raise NotImplementedError()
