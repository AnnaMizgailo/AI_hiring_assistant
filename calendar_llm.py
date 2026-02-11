from typing import Any, Dict, List
import uuid

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def get_google_oauth_url(store, recruiter_id: str) -> str:
    raise NotImplementedError()

def handle_google_callback(store, recruiter_id: str, code: str) -> Dict[str, Any]:
    raise NotImplementedError()

def get_freebusy(store, recruiter_id: str, time_min: str, time_max: str) -> List[Dict[str, Any]]:
    raise NotImplementedError()

def build_candidate_slots(free_windows: List[Dict[str, Any]], slot_minutes: int = 30, max_slots: int = 8) -> List[Dict[str, Any]]:
    raise NotImplementedError()

def create_calendar_event(store, recruiter_id: str, start_dt: str, end_dt: str, candidate_name: str, description: str) -> Dict[str, Any]:
    raise NotImplementedError()

def get_available_slots(store, job_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError()

def book_slot(store, application_id: str, slot_id: str) -> Dict[str, Any]:
    raise NotImplementedError()

def generate_screening_summary(store, application_id: str) -> str:
    raise NotImplementedError()

def generate_feedback(store, application_id: str) -> str:
    raise NotImplementedError()
